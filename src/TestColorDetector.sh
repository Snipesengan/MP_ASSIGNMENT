file="res/SetD"
file="res/SetC"
file="SampleData-20180826/20180826/SetC"
file="20180920/SetB"
file="SampleData-20180826/20180826/SetD"
file="20180920/SetD"
file="20180920/SetC"
file="SampleData-20180826/20180826/SetB"
file="res/SetA"


for image in $(ls $file); do
    echo $image
    python3 src/HLD_Pipeline.py $file'/'$image 
    echo
done
