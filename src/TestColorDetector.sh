file="res/SetA"
file="res/SetD"
file="SampleData-20180826/20180826/SetC"
file="res/SetC"
file="SampleData-20180826/20180826/SetB"
file="SampleData-20180826/20180826/SetD"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image
    echo
done
