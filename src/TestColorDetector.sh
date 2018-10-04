file="res/SetB"
file="20180920/SetB"
file="res/SetC"
file="20180920/SetC"
file="20180920/SetD"
file="res/Test_Data/SampleData-20180826/20180826/SetC"
file="res/Test_Data/SetD"
file="res/Test_Data/SetA"
file="res/Test_Data/SampleData-20180826/20180826/SetD"
file="res/Test_Data/SampleData-20180826/20180826/SetB"
file="res/Test_Data/check/SetA"
file="res/Test_Data/check/SetB"
file="res/Test_Data/check/SetC"
file="res/Test_Data/check/SetD"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image -display
    echo
done
