import torch
import time
import os
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from thop import profile
import re


def show_dataset_prev(train_loader, test_loader, val_loader=None, num_images=3, num_classes=1):
    images_shown = 0

    # Cria um iterador para o val_loader se for fornecido
    val_iter = iter(val_loader) if val_loader is not None else None

    for (images_train, masks_train), (images_test, masks_test) in zip(train_loader, test_loader):
        if val_iter:
            try:
                images_val, masks_val = next(val_iter)
            except StopIteration:
                break  # Encerra se o val_loader acabar

        for i in range(images_train.size(0)):
            if images_shown >= num_images:
                break

            def process_image_mask(img_tensor, mask_tensor):
                img_tensor = img_tensor.cpu()
                img = img_tensor.permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.squeeze(0).numpy()
                img = img * 0.5 + 0.5
                mask = mask_tensor.cpu().squeeze().numpy()
                return img, mask

            img_train, mask_train = process_image_mask(images_train[i], masks_train[i])
            img_test, mask_test = process_image_mask(images_test[i], masks_test[i])
            
            if val_iter:
                img_val, mask_val = process_image_mask(images_val[i], masks_val[i])

            # Define número de colunas com base no val_loader
            n_cols = 6 if val_iter else 4
            fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 4))

            axs[0].imshow(img_train)
            axs[0].set_title("Imagem Treino")
            axs[1].imshow(mask_train, cmap='viridis', vmin=0, vmax=num_classes)
            axs[1].set_title("Máscara Treino")

            axs[2].imshow(img_test)
            axs[2].set_title("Imagem Teste")
            axs[3].imshow(mask_test, cmap='viridis', vmin=0, vmax=num_classes)
            axs[3].set_title("Máscara Teste")

            if val_iter:
                axs[4].imshow(img_val)
                axs[4].set_title("Imagem Val")
                axs[5].imshow(mask_val, cmap='viridis', vmin=0, vmax=num_classes)
                axs[5].set_title("Máscara Val")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

            images_shown += 1

        if images_shown >= num_images:
            break

    
def verificar_mascara_multiclasse(mascara, num_classes):
    if len(mascara.shape) != 2:
        print("Formato inválido: a máscara deve ser [H, W]")
    else:
        print("Formato ok")

    valores = np.unique(mascara)
    if not np.all(np.equal(valores, valores.astype(int))):
        print("A máscara contém valores não inteiros")
    else:
        print(f"Valores únicos: {valores}")

    if mascara.dtype not in [np.uint8, np.int32, np.int64]:
        print(f"Tipo inválido: {mascara.dtype}")
    else:
        print("Tipo de dado ok")

    if valores.min() < 0 or valores.max() >= num_classes:
        print(f"Valores fora do intervalo esperado [0, {num_classes - 1}]")
    else:
        print("Intervalo de valores está correto")


def beep():
    os.system('powershell.exe -Command "[console]::beep(500,400); [console]::beep(500,400)"')


def count_trainable_parameters(model, format=False):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if format:
        return f"{total:,}".replace(",", ".")
    return total


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def measure_inference_speed(model, test_loader, measure_cpu_speed=True):
    model.eval()

    # Pega apenas o primeiro batch do test_loader
    inputs, _ = next(iter(test_loader))

    results = {}
    if measure_cpu_speed:
        devices = ['cuda', 'cpu']
    else:
        devices = ['cuda']

    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            results['gpu'] = (None, None)
            continue

        model.to(device)
        inputs_device = inputs.to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(inputs_device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        # Mede o tempo de forward puro (100 execuções do mesmo batch)
        steps = 1
        with torch.no_grad():
            for _ in range(steps):
                _ = model(inputs_device)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_image = total_time / (steps * inputs_device.size(0))
        time_per_image = f"{avg_time_per_image * 1000:.3f} ms"
        fps = 1.0 / avg_time_per_image

        results['gpu' if device == 'cuda' else 'cpu'] = (fps, time_per_image)

    fps_gpu, time_per_image_gpu = results.get('gpu', (None, None))
    fps_cpu, time_per_image_cpu = results.get('cpu', (None, None))

    return fps_gpu, time_per_image_gpu, fps_cpu, time_per_image_cpu


def compile_xls_best_results(input_dir, output_file="resultado.xlsx"):
    linhas = []
    modelo_atual = None
    bloco_linhas = []

    files = os.listdir(input_dir)
    files.sort()

    def adicionar_bloco(linhas, bloco_linhas, modelo):
        if not bloco_linhas:
            return
        # Converte bloco em DataFrame
        df_bloco = pd.DataFrame(bloco_linhas)

        # Calcula médias (apenas para colunas Dice e FPS se existirem)
        media_row = {"arquivo": f"{modelo}-MÉDIA"}
        if "dice" in df_bloco.columns:
            media_row["dice"] = df_bloco["dice"].mean()
        if "FPS" in df_bloco.columns:
            media_row["FPS"] = df_bloco["FPS"].mean()

        # Adiciona bloco + linha de média + linha em branco
        linhas.extend(df_bloco.to_dict("records"))
        linhas.append(media_row)   # 1ª linha em branco com média
        linhas.append({})          # 2ª linha em branco

    for file in files:
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_dir, file)

            try:
                # Nome base do modelo (tudo antes do último "-número")
                match = re.match(r"(.+)-\d+$", file.replace("-epochs300.xlsx", ""))
                if match:
                    modelo = match.group(1)
                else:
                    modelo = file.replace(".xlsx", "")

                # Lê a aba val_history
                val_history = pd.read_excel(file_path, sheet_name="val_history")

                # Pega a linha com maior valor na coluna "dice"
                best_row = val_history.loc[val_history["dice"].idxmax()].copy()

                # Lê a aba model_info
                model_info = pd.read_excel(file_path, sheet_name="model_info")

                # Procura a coluna FPS
                fps_value = model_info["FPS"].iloc[0] if "FPS" in model_info.columns else None

                # Converte a linha em Series e adiciona FPS
                best_row["FPS"] = fps_value
                best_row = pd.Series({"arquivo": file.replace(".xlsx", ""), **best_row.to_dict()})

                # Se mudou de modelo, fecha o bloco anterior
                if modelo_atual is not None and modelo != modelo_atual:
                    adicionar_bloco(linhas, bloco_linhas, modelo_atual)
                    bloco_linhas = []

                # Atualiza modelo atual
                modelo_atual = modelo

                # Adiciona linha ao bloco
                bloco_linhas.append(best_row)

            except Exception as e:
                print(f"Erro ao processar {file}: {e}")

    # Finaliza último bloco
    if bloco_linhas:
        adicionar_bloco(linhas, bloco_linhas, modelo_atual)

    # Junta todas as linhas em um DataFrame
    df_final = pd.DataFrame(linhas)

    # Salva em Excel
    df_final.to_excel(output_file, index=False)
    print(f"Arquivo salvo em: {output_file}")



def get_next_run_dir(base_dir='./tb_profiler', prefix='run_'):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    existing_ids = []
    for d in existing:
        try:
            existing_ids.append(int(d.replace(prefix, '')))
        except ValueError:
            continue
    next_id = max(existing_ids, default=0) + 1
    return os.path.join(base_dir, f'{prefix}{next_id}')

def run_profiler(model, data_loader, model_name='model', num_steps=1):
    device = 'cuda'
    logdir = get_next_run_dir('./tb_profiler_streams', prefix=f'{model_name}_run_')

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for i, (images, masks) in enumerate(data_loader):
                if i >= num_steps:
                    break
                images = images.to(device)
                masks  = masks.to(device)
                model.to(device)

                with record_function("model_inference"):
                    _ = model(images)
                #torch.cuda.synchronize()

    print(f"Profiling results saved to: {logdir}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def get_flops_gflops(model, input_size=(1, 3, 224, 224), device='cuda'):
    dummy_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    gflops = flops / 1e9  # converte para GFLOPs
    return gflops