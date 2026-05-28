from dataloaders.sysu_dataloader import SYSUDataset, get_default_transforms

dataset = SYSUDataset(
    root="dataset",
    mode="train",
    modality="ir",
    transform=get_default_transforms(train=True)
)

print("Total samples:", len(dataset))
