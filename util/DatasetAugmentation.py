#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#jupyter nbconvert --to script DatasetAugmentation.ipynb

import os
import cv2
import random
import shutil
import albumentations as A
from tqdm import tqdm
from PIL import Image
import numpy as np

images_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

# -----------------------------
# Funções auxiliares
# -----------------------------

def copy_and_fix(img_src_dir, mask_src_dir, img_out_dir, mask_out_dir, selected_files=None, 
                 function_to_apply_to_masks=None, mask_suffix=''):
    """
    Copia imagens e máscaras de img_src_dir/mask_src_dir para img_out_dir/mask_out_dir.
    Se selected_files for None, copia todos os arquivos da pasta de origem.
    Aceita máscaras com qualquer extensão (.png, .jpg, .jpeg .bmp).
    """
    if not os.path.exists(img_src_dir) or not os.path.exists(mask_src_dir):
        return 0  # nada para copiar

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    if selected_files is None:
        files = sorted([f for f in os.listdir(img_src_dir) if f.lower().endswith(images_extensions)])
    else:
        files = selected_files

    count = 0
    for f in tqdm(files, desc=f"Copiando {os.path.basename(img_out_dir)}"):
        img_src = os.path.join(img_src_dir, f)
        base = os.path.splitext(f)[0]

        # procurar a máscara com qualquer extensão
        possible_mask_paths = [
            os.path.join(mask_src_dir, base + mask_suffix + ext)
            for ext in images_extensions
        ]
        mask_src = next((p for p in possible_mask_paths if os.path.exists(p)), None)

        if mask_src and os.path.exists(img_src):
            # Copia imagem
            shutil.copy(img_src, os.path.join(img_out_dir, base + ".png"))
            # Copia máscara, convertendo pra PNG
            mask = np.array(Image.open(mask_src).convert("L"))
            if function_to_apply_to_masks is not None:
                #Aplica a funcao de correcao
                mask = function_to_apply_to_masks(mask)
            Image.fromarray(mask).save(os.path.join(mask_out_dir, base + ".png"))
            count += 1
    return count



# -----------------------------
# Augmentation para treino
# -----------------------------
def augment_train_images(image_list, image_dir, mask_dir, output_image_dir, output_mask_dir, transforms, N, 
                         function_to_apply_to_masks=None, mask_suffix=''):
    for img_name in tqdm(image_list, desc="Aumentando imagens de treino"):
        # Nome base sem extensão
        base_name = os.path.splitext(img_name)[0]

        # Caminhos possíveis (.png, .jpg, .jpeg)
        possible_img_paths = [
            os.path.join(image_dir, base_name + ext) for ext in images_extensions
        ]
        possible_mask_paths = [
            os.path.join(mask_dir, base_name + mask_suffix + ext) for ext in images_extensions
        ]

        # Escolhe o primeiro arquivo que existir
        img_path = next((p for p in possible_img_paths if os.path.exists(p)), None)
        mask_path = next((p for p in possible_mask_paths if os.path.exists(p)), None)

        if img_path is None or mask_path is None:
            print(f"[AVISO] Arquivo não encontrado para {base_name}. Pulando.")
            print(possible_mask_paths)
            continue

        # Lê imagem e máscara
        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if function_to_apply_to_masks is not None:
            mask  = function_to_apply_to_masks(mask)


        if image is None or mask is None:
            print(f"[AVISO] Falha ao ler {base_name}. Pulando.")
            continue

        # Transforma e salva versão original como PNG
        orig = transforms(image=image, mask=mask)
        cv2.imwrite(os.path.join(output_image_dir, f"{base_name}_orig.png"), orig['image'])
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_orig.png"), orig['mask'])

        # Gera augmentations
        for i in range(N):
            aug = transforms(image=image, mask=mask)
            cv2.imwrite(os.path.join(output_image_dir, f"{base_name}_aug{i}.png"), aug['image'])
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_aug{i}.png"), aug['mask'])



def augment_dataset(N, num_to_valid, num_to_test,  
                    orig_train_img_dir, orig_train_mask_dir,
                    orig_valid_img_dir, orig_valid_mask_dir,
                    orig_test_img_dir, orig_test_mask_dir,
                    output_base,
                    transforms,
                    function_to_apply_to_masks=None,
                    mask_suffix=''):

    output_dirs = {
        'train_images': os.path.join(output_base, 'images/train'),
        'train_labels': os.path.join(output_base, 'labels/train'),
        'valid_images': os.path.join(output_base, 'images/valid'),
        'valid_labels': os.path.join(output_base, 'labels/valid'),
        'test_images':  os.path.join(output_base, 'images/test'),
        'test_labels':  os.path.join(output_base, 'labels/test'),
    }

    # -----------------------------
    # Selecionar imagens para splits
    # -----------------------------
    all_images = sorted([f for f in os.listdir(orig_train_img_dir) if f.lower().endswith(images_extensions)])
    total_imgs = len(all_images)

    if num_to_valid + num_to_test >= total_imgs:
        print("num_to_valid:",num_to_valid, "num_to_test:", num_to_test, "total_imgs:",total_imgs)
        raise ValueError("Quantidade de imagens para valid+test é maior ou igual ao total disponível.")

    # Amostras aleatórias (sem modificar o diretório original)
    selected_test = set(random.sample(all_images, num_to_test))
    remaining = [f for f in all_images if f not in selected_test]

    selected_valid = set(random.sample(remaining, num_to_valid))
    remaining = [f for f in remaining if f not in selected_valid]

    train_images = remaining

    # -----------------------------
    # Estimativa total de imagens geradas
    # -----------------------------
    train_total = len(train_images)
    total_output = train_total * (N + 1)

    print(f"Imagens totais no dataset original: {total_imgs}")
    print(f"→ Treino: {len(train_images)}")
    print(f"→ Validação (do treino): {len(selected_valid)}")
    print(f"→ Teste (do treino): {len(selected_test)}")
    print(f"\nCom N={N}, total de imagens geradas no treino será: {total_output}")

    choice = input("Deseja prosseguir? (y/n): ").strip().lower()
    if choice not in ('y', 's'):
        print("Processo cancelado.")
        raise SystemExit

    # -----------------------------
    # Criação das pastas
    # -----------------------------
    if any(os.path.exists(d) for d in output_dirs.values()):
        print("O diretório de saída já existe. Abortando para evitar sobrescrita.")
        raise SystemExit
    else:
        for d in output_dirs.values():
            os.makedirs(d, exist_ok=True)





    # -----------------------------
    # Copiar pastas existentes de valid/test
    # -----------------------------
    count_valid_existing = copy_and_fix(orig_valid_img_dir, orig_valid_mask_dir,
                                        output_dirs['valid_images'], output_dirs['valid_labels'],
                                        function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
    count_test_existing  = copy_and_fix(orig_test_img_dir, orig_test_mask_dir,
                                        output_dirs['test_images'], output_dirs['test_labels'],
                                        function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)

    print(f"→ {count_valid_existing} imagens copiadas da pasta valid original.")
    print(f"→ {count_test_existing} imagens copiadas da pasta test original.")

    # -----------------------------
    # Mover imagens do train para valid/test se num_to_valid/num_to_test > 0
    # -----------------------------
    # Valid
    if num_to_valid > 0:
        selected_valid = list(selected_valid)
        copied = copy_and_fix(orig_train_img_dir, orig_train_mask_dir,
                            output_dirs['valid_images'], output_dirs['valid_labels'], selected_valid,
                            function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
        print(f"→ {copied} imagens copiadas do train para valid.")

    # Test
    if num_to_test > 0:
        selected_test = list(selected_test)
        copied = copy_and_fix(orig_train_img_dir, orig_train_mask_dir,
                            output_dirs['test_images'], output_dirs['test_labels'], selected_test,
                            function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)
        print(f"→ {copied} imagens copiadas do train para test.")



    # -----------------------------
    # Processar o conjunto de treino
    # -----------------------------
    augment_train_images(train_images, orig_train_img_dir, orig_train_mask_dir,
                    output_dirs['train_images'], output_dirs['train_labels'],
                    transforms, N,
                    function_to_apply_to_masks=function_to_apply_to_masks, mask_suffix=mask_suffix)

    # -----------------------------
    # Relatório final
    # -----------------------------
    def count_images_in_dir(directory):
        return len([f for f in os.listdir(directory) if f.lower().endswith(images_extensions)])

    print("\nResumo final:")
    for key, path in output_dirs.items():
        print(f"{key}: {count_images_in_dir(path)} arquivos")


# In[6]:


if __name__ == "__main__":

    #Uma classe para simular o arquivo config presente em cada dataset
    class Config:
        pass

    config = Config()
    config.dataset_path          = "/mnt/TUDAO/0PequeNet/DatasetAugmentationTest/AugmentationTest"
    config.original_dataset_path = "/mnt/TUDAO/0PequeNet/medetec/datasets/Medetec_foot_ulcer_224"
    config.dataset_resolution    = 224

    # -----------------------------
    # Parâmetros
    # -----------------------------
    N = 2  # número de aumentações
    num_to_valid = 5  # número de imagens a mover do train para valid
    num_to_test  = 5  # número de imagens a mover do train para test
    target_size  = (config.dataset_resolution, config.dataset_resolution)
    random.seed(42)

    # -----------------------------
    # Caminhos de entrada e saída
    # -----------------------------
    orig_train_img_dir  = os.path.join(config.original_dataset_path, 'train/images')
    orig_train_mask_dir = os.path.join(config.original_dataset_path, 'train/labels')
    orig_valid_img_dir  = os.path.join(config.original_dataset_path, 'valid/images')
    orig_valid_mask_dir = os.path.join(config.original_dataset_path, 'valid/labels')
    orig_test_img_dir   = os.path.join(config.original_dataset_path, 'test/images')
    orig_test_mask_dir  = os.path.join(config.original_dataset_path, 'test/labels')


    output_base = config.dataset_path

    # -----------------------------
    # Transformações de aumento
    # -----------------------------
    transforms = A.Compose([
        A.Resize(*target_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.2),
        A.GaussianBlur(p=0.3),
        A.GridDistortion(p=0.2),
    ])

    augment_dataset(N, num_to_valid, num_to_test,
                    orig_train_img_dir, orig_train_mask_dir,
                    orig_valid_img_dir, orig_valid_mask_dir,
                    orig_test_img_dir, orig_test_mask_dir,
                    output_base,
                    transforms)


