file="20180920/SetC"
file="res/Test_Data/SetD"
file="res/Test_Data/SampleData-20180826/20180826/SetD"
file="res/Test_Data/check/SetA"
file="res/Test_Data/check/SetB"
file="res/Test_Data/check/SetC"
file="res/Test_Data/check/SetD"
file="res/Test_Data/SetD"
file="res/Test_Data/SetC"
file="res/Test_Data/SampleData-20180826/20180826/SetC"
file="res/Test_Data/SampleData-20180826/20180826/SetB"
file="res/Test_Data/20180920/SetB"
file="res/Test_Data/20180920/SetD"
file="res/Test_Data/SetB"
file="res/Test_Data/SetA"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image -display
    echo
done
