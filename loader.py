import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import tensor
from PIL import Image

TRAIN_PATH = "DATASET/OUT/train"
TEST_PATH = "DATASET/OUT/test"
VALID_PATH = "DATASET/OUT/valid"

transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
        transforms.transforms.Lambda(lambda x: x.clamp(0, 1)),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, BASE_DIR, transform=None):
        self.base_dir = BASE_DIR
        self.image_dir = os.path.join(BASE_DIR, "images")
        self.mask_dir = os.path.join(BASE_DIR, "masks")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB").convert("L")
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


train_dataset = CustomDataset(TRAIN_PATH, transform)
test_dataset = CustomDataset(TEST_PATH, transform)
valid_dataset = CustomDataset(VALID_PATH, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
