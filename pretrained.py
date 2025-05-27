import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForWholeWordMask,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import logging

from config import AdaptationConfig
from sentence_reader import SentenceReader

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class ElectraDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_input_ids = torch.stack([ex['input_ids'] for ex in examples])
        batch_attention_mask = torch.stack(
            [ex['attention_mask'] for ex in examples])

        original_input_ids = batch_input_ids.clone()

        generator_labels = original_input_ids.clone()

        probability_matrix = torch.full(
            original_input_ids.shape, self.mlm_probability, device=original_input_ids.device)

        special_tokens_mask_list = []
        for val in original_input_ids.tolist():
            special_tokens_mask_list.append(
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
            )
        special_tokens_mask_t = torch.tensor(
            special_tokens_mask_list, dtype=torch.bool, device=original_input_ids.device)
        probability_matrix.masked_fill_(special_tokens_mask_t, value=0.0)

        mlm_masked_indices = torch.bernoulli(probability_matrix).bool()

        # -100 is the ignore index for CrossEntropyLoss
        generator_labels[~mlm_masked_indices] = -100

        generator_input_ids_for_model = original_input_ids.clone()

        # 80% of the time, replace with [MASK]
        indices_replaced_mask = torch.bernoulli(torch.full(
            original_input_ids.shape, 0.8, device=original_input_ids.device)).bool() & mlm_masked_indices
        generator_input_ids_for_model[indices_replaced_mask] = self.tokenizer.mask_token_id

        # 10% of the time, replace with a random token (the remaining 10% is original token, already handled by clone)
        indices_random_token = torch.bernoulli(torch.full(
            # 0.5 of the remaining 20% -> 10% of total
            original_input_ids.shape, 0.5, device=original_input_ids.device)).bool() & mlm_masked_indices & ~indices_replaced_mask
        random_words = torch.randint(len(
            self.tokenizer), original_input_ids.shape, dtype=torch.long, device=original_input_ids.device)
        generator_input_ids_for_model[indices_random_token] = random_words[indices_random_token]

        # The remaining 10% of mlm_masked_indices will keep their original tokens

        return {
            # For discriminator, and to create labels
            "original_input_ids": original_input_ids,
            "generator_input_ids": generator_input_ids_for_model,  # Input to generator
            "generator_labels": generator_labels,  # Labels for generator (MLM)
            # Boolean mask indicating which tokens were originally selected for MLM
            "mlm_masked_indices": mlm_masked_indices,
            "attention_mask": batch_attention_mask,
        }


def electra_train(config: AdaptationConfig):
    """训练函数"""
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.discriminator_model_name_or_path)

    # 加载Generator和Discriminator模型
    generator = AutoModelForMaskedLM.from_pretrained(
        config.generator_model_name_or_path)
    discriminator = AutoModelForTokenClassification.from_pretrained(
        # 0 for original, 1 for replaced
        config.discriminator_model_name_or_path, num_labels=2)

    # 确保输出目录存在
    if not os.path.exists(config.adapted_model_dir):
        os.makedirs(config.adapted_model_dir)
    logger.info(f"模型将保存到: {config.adapted_model_dir}")

    # 加载领域语料
    sentence_reader = SentenceReader(config.corpus_file)
    texts = sentence_reader.read()
    logger.info(f"加载了 {len(texts)} 条语料 from {config.corpus_file}")

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=config.max_length)

    # 创建Electra数据整理器
    data_collator = ElectraDataCollator(
        tokenizer=tokenizer,
        mlm_probability=config.mask_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )

    # 准备训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    logger.info(f"训练设备: {device}")

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(generator.named_parameters()) + list(discriminator.named_parameters()) if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in list(generator.named_parameters()) + list(discriminator.named_parameters()) if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=config.learning_rate, eps=config.adam_epsilon)

    # 学习率调度器
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # 训练循环
    global_step = 0
    logger.info(
        f"开始ELECTRA训练 (Generator + Discriminator) for {config.num_epochs} epochs")
    config.adapted_model_dir = os.path.join(
            "pretrained", f"{config.discriminator_model_name_or_path.replace('/', '_')}_electra_adapted_ep{config.num_epochs}_seed{config.seed}"
        )
    for epoch in range(config.num_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        epoch_total_loss = 0
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        for step, batch in enumerate(epoch_iterator):
            generator.train()
            discriminator.train()

            # 将数据移到设备上
            original_input_ids = batch['original_input_ids'].to(device)
            generator_input_ids = batch['generator_input_ids'].to(device)
            generator_labels = batch['generator_labels'].to(device)
            mlm_masked_indices = batch['mlm_masked_indices'].to(
                device)  # boolean mask
            attention_mask = batch['attention_mask'].to(device)

            # 1. Generator Forward Pass (MLM task)
            gen_outputs = generator(
                input_ids=generator_input_ids,
                attention_mask=attention_mask,
                labels=generator_labels
            )
            loss_gen = gen_outputs.loss
            gen_logits = gen_outputs.logits

            # 2. Construct Discriminator Input using Generator's samples
            discriminator_input_ids = original_input_ids.clone()
            with torch.no_grad():
                sampled_tokens = torch.argmax(gen_logits, dim=-1)
                discriminator_input_ids[mlm_masked_indices] = sampled_tokens[mlm_masked_indices]
                actual_discriminator_labels = (
                    discriminator_input_ids != original_input_ids).long()

            # 3. Discriminator Forward Pass & Loss
            disc_outputs = discriminator(
                input_ids=discriminator_input_ids,
                attention_mask=attention_mask,
                labels=actual_discriminator_labels
            )
            loss_disc = disc_outputs.loss

            # 4. Total Loss
            total_loss = config.generator_loss_weight * loss_gen + \
                config.discriminator_loss_weight * loss_disc

            epoch_total_loss += total_loss.item()
            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_disc.item()

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({
                "total_loss": total_loss.item(),
                "gen_loss": loss_gen.item(),
                "disc_loss": loss_disc.item()
            })

        # 保存每个epoch的模型
        avg_epoch_total_loss = epoch_total_loss / \
            len(dataloader) if len(dataloader) > 0 else 0
        avg_epoch_gen_loss = epoch_gen_loss / \
            len(dataloader) if len(dataloader) > 0 else 0
        avg_epoch_disc_loss = epoch_disc_loss / \
            len(dataloader) if len(dataloader) > 0 else 0
        logger.info(
            f"Epoch {epoch+1} 平均损失: Total={avg_epoch_total_loss:.4f}, Gen={avg_epoch_gen_loss:.4f}, Disc={avg_epoch_disc_loss:.4f}")

        epoch_dir = os.path.join(config.adapted_model_dir, f"epoch-{epoch+1}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        gen_save_dir = os.path.join(epoch_dir, "generator")
        disc_save_dir = os.path.join(epoch_dir, "discriminator")
        if not os.path.exists(gen_save_dir):
            os.makedirs(gen_save_dir)
        if not os.path.exists(disc_save_dir):
            os.makedirs(disc_save_dir)

        generator.save_pretrained(gen_save_dir)
        discriminator.save_pretrained(disc_save_dir)
        # Tokenizer saved per epoch with models
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"保存epoch {epoch+1}的模型到 {epoch_dir}")

    # 保存最终模型
    final_gen_save_dir = os.path.join(
        config.adapted_model_dir, "generator_final")
    final_disc_save_dir = os.path.join(
        config.adapted_model_dir)
    if not os.path.exists(final_gen_save_dir):
        os.makedirs(final_gen_save_dir)
    if not os.path.exists(final_disc_save_dir):
        os.makedirs(final_disc_save_dir)
    generator.save_pretrained(final_gen_save_dir)
    discriminator.save_pretrained(final_disc_save_dir)
    # Save tokenizer in the main output dir for the final model
    tokenizer.save_pretrained(config.adapted_model_dir)
    logger.info(f"训练完成，最终模型保存到 {config.adapted_model_dir}")

    return config.adapted_model_dir


def mlm_train(config: AdaptationConfig):
    """领域适配预训练函数"""
    logger.info(f"Starting domain adaptation with parameters: {locals()}")
    config.adapted_model_dir = os.path.join(
        "pretrained", f"{config.model_name.replace('/', '_')}_adapted_ep{config.num_epochs}_seed{config.seed}"
    )
    config.adapted_model_path = os.path.join(
        config.adapted_model_dir, "model.pt" # .save_pretrained saves more than just one file
    )

    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        logger.info("CUDA is available. Using GPU for adaptation.")
    else:
        logger.info("CUDA not available. Using CPU for adaptation.")

    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    except Exception as e:
        logger.error(
            f"Error loading model or tokenizer from {config.model_name}: {e}")
        return None

    # 确保输出目录存在
    if not os.path.exists(config.adapted_model_dir):
        os.makedirs(config.adapted_model_dir)
        logger.info(f"Created output directory: {config.adapted_model_dir}")

    # 加载领域语料
    sentence_reader = SentenceReader()
    texts = sentence_reader.read_corpus(config.corpus_file)
    logger.info(f"加载了 {len(texts)} 条语料 from {config.corpus_file}")
    if not texts:
        logger.error("No texts loaded from corpus. Aborting adaptation.")
        return None

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=config.max_length)

    # 创建数据整理器用于全词掩码
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mask_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # 准备训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if not any(pg["params"] for pg in optimizer_grouped_parameters):
        logger.error(
            "No parameters to optimize. Check model and no_decay list.")
        return None

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=config.learning_rate, eps=config.adam_epsilon)

    # 学习率调度器
    total_steps = len(dataloader) * config.num_epochs
    scheduler = None
    if total_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
        )
    else:
        logger.warning(
            "Total steps is 0. No training will occur. Check data and batch size.")
        # Save the original model if no training steps
        model.save_pretrained(config.adapted_model_dir)
        tokenizer.save_pretrained(config.adapted_model_dir)
        logger.info(
            f"No training steps. Original model saved to {config.adapted_model_dir}")
        return config.adapted_model_dir

    # 训练循环
    global_step = 0
    model.zero_grad()  # Clear gradients before starting training

    logger.info("开始训练 (domain adaptation)")
    for epoch in range(config.num_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Adaptation)")
        epoch_loss = 0
        batch_count = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()

            # 将数据移到设备上
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
            except AttributeError:
                logger.error(
                    f"Error moving batch to device. Batch type: {type(batch)}, keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
                continue  # Skip this batch

            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

            if loss is None:
                logger.warning(
                    f"Loss is None at epoch {epoch+1}, step {step}. Skipping batch.")
                continue

            epoch_loss += loss.item()
            batch_count += 1

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({"loss": loss.item()})

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(
                f"Epoch {epoch+1} (Adaptation) 平均损失: {avg_epoch_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} (Adaptation) had no batches.")

    # 保存最终模型
    model.save_pretrained(config.adapted_model_dir)
    tokenizer.save_pretrained(config.adapted_model_dir)
    logger.info(f"领域适配预训练完成，最终模型保存到 {config.adapted_model_dir}")

    return config.adapted_model_dir


if __name__ == "__main__":
    # Load configuration from YAML file using AdaptationConfig
    adaptation_config = AdaptationConfig('pretrained.yaml')

    # Ensure the output directory from config is used/created by the train function
    # The train function now handles creation of config.adapted_model_dir

    logger.info(
        f"Starting domain adaptation pre-training with config: {adaptation_config.__dict__}")

    output_directory = mlm_train(adaptation_config)

    logger.info(f"领域适配预训练完成，模型保存在 {output_directory}")
