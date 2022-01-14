mkdir -p data/train
mkdir -p data/val

cp -r train_data/beth/* data/train/
cp -r train_data/partners/* data/train/

cp -r val_data/* data/val/