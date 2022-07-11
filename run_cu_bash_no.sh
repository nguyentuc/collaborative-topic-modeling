#! /bin/bash
dataname=data_citeulike
data_path="../../Data/$dataname"

num_factors=100
RES=results
if [ ! -d "$RES" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $RES
fi

#root_path=../data/arxiv/cv
#num_factors=100
for l in 0
do
	for drate in 0.8 0.9 2
	do
		./ctr --directory $dataname --user $data_path/users_train.dat --item $data_path/items_train.dat --a 1 --b 1 --lambda_u 0.01 --lambda_v 0.01 \
		--num_factors $num_factors --max_iter 100 --learning_rate $drate
		echo "Eval $dataname rate $drate"
		python evalb.py -d $dataname -n all -r $dataname-$drate
	done
done