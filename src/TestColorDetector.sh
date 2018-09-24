file="20180920/SetD"
file="res/SetA"
file="20180920/SetC"
file="SampleData-20180826/20180826/SetD"
file="SampleData-20180826/20180826/SetB"
file="20180920/SetB"
file="res/SetB"
file="SampleData-20180826/20180826/SetC"
file="res/SetC"
file="res/SetD"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image -display
    echo
done
