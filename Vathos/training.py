from tqdm.notebook import tqdm
from Vathos.blocks import *


def symbolic_1d_ar_target_train(model, train_loader, val_loader, optimizer, scheduler, criterion,
          device, NUM_EPOCHS=100, use_amp=False, CLIP_NORM=1.0, div=1):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = len(train_loader) // div

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Train", total=steps_per_epoch)
        steps = 0

        for batch_x, batch_tgt in pbar:
            batch_x = batch_x.to(device)
            steps += 1

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)  # [B,S,V]
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch_x[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            running_loss += loss.item()
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'lr': lr_now})

            if steps >= steps_per_epoch:
                break

        model.eval()
        val_loss = 0.0
        # with torch.no_grad():
        if True:
            for batch_x, batch_tgt, composer_idx in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Val"):
                batch_x = batch_x.to(device).requires_grad_(False)

                logits = model(batch_x)
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch_x[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                if isinstance(model, VathosModel):
                    model.module.register_loss(loss.item())

        val_avg = val_loss / len(val_loader)
        if isinstance(model, VathosModel):
            model.module.register_epoch()
        print(f"Epoch {epoch + 1} val avg loss: {val_avg:.4f}")


def symbolic_1d_ar_input_ids_train(model, train_loader, val_loader, optimizer, scheduler, criterion,
                                      device, NUM_EPOCHS=100, use_amp=False, CLIP_NORM=1.0, spe=1000):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = spe

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Train", total=steps_per_epoch)
        steps = 0

        for batch_x in pbar:
            batch_x = batch_x['input_ids'].to(device)
            steps += 1

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)  # [B,S,V]
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch_x[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'lr': lr_now})

            if steps >= steps_per_epoch:
                break

        model.eval()
        val_loss = 0.0
        # with torch.no_grad():
        if True:
            for batch_x in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Val"):
                batch_x = batch_x['input_ids'].to(device).requires_grad_(False)

                logits = model(batch_x)
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch_x[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                model.module.register_loss(loss.item())

        val_avg = val_loss / len(val_loader)
        model.module.register_epoch()
        print(f"Epoch {epoch + 1} val avg loss: {val_avg:.4f}")


symbolic_1d_ar_pretokenized_train(model, train_loader, val_loader, optimizer, scheduler=scheduler, criterion=criterion,
                                  device=device, NUM_EPOCHS=100, use_amp=use_amp)


