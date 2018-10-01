file="res/SetB"
file="SampleData-20180826/20180826/SetB"
file="20180920/SetB"
file="res/SetC"
file="20180920/SetC"
file="20180920/SetD"
file="SampleData-20180826/20180826/SetD"
file="res/Test_Data/SampleData-20180826/20180826/SetC"
file="res/Test_Data/SetD"
file="res/Test_Data/SetA"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image -display
    echo
done
