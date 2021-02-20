wget -P ./data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget -P ./data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
wget -P ./data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip 
mkdir ./data/GTSRB;
mkdir ./data/GTSRB/Train;
mkdir ./data/GTSRB/Test;
mkdir ./data/temps;
unzip ./data/GTSRB_Final_Training_Images.zip -d ./data/temps/Train;
unzip ./data/GTSRB_Final_Test_Images.zip -d ./data/temps/Test;
mv ./data/temps/Train/GTSRB/Final_Training/Images/* ./data/GTSRB/Train;
mv ./data/temps/Test/GTSRB/Final_Test/Images/* ./data/GTSRB/Test;
unzip ./data/GTSRB_Final_Test_GT.zip -d ./data/GTSRB/Test/;
rm -r ./data/temps;
rm ./data/*.zip;
echo "Download Completed";
