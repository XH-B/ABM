export CUDA_VISIBLE_DEVICES=0

path='./ABM.pkl' 

dataset=$1
branch=$2

if [ $branch == "L2R" ]
then
   idx_decoder=1 
elif [ $branch == "R2L" ] 
then
   idx_decoder=2
fi

echo $idx_decoder


for year in $dataset
do
python test.py \
	-k 10 \
	-model_path $path \
	-dictionary_target "./data/dictionary.txt" \
	-test_dataset "./data/offline-$year-test.pkl" \
	-label "./data/test-caption-$year.txt" \
	-saveto "./result/$year-branch_$branch.txt" \
	-output "./result/test_$dataset-branch_$branch.wer" \
	-idx_decoder $idx_decoder \
	
done


