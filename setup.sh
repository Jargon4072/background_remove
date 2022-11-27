wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

mkdir xception_model
tar xvzf deeplabv3_pascal_train_aug_2018_01_04.tar.gz -C xception_model --strip=1

rm deeplabv3_pascal_train_aug_2018_01_04.tar.gz