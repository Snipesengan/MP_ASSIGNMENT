for image in $(ls SampleData-20180826/20180826/SetD); do
    echo $image
    python3 src/HLD_Pipeline.py 'SampleData-20180826/20180826/SetD/'$image
    echo
done
