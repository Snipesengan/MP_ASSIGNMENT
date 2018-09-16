file="SampleData-20180826/20180826/SetB"
file="SampleData-20180826/20180826/SetD"
file="res/SetD"
file="res/SetA"
file="res/SetC"
file="SampleData-20180826/20180826/SetC"

for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image 
    echo
done
