#! /bin/bash
dataname=Yelp
data_path="../../Data/$dataname"

num_factors=100
RES=results
if [ ! -d "$RES" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $RES
fi

#root_path=../data/arxiv/cv
#num_factors=100


for l in 0 1 2 3 4
do
	for drate in 2 0.9 0.8 0.7 0.6 0.5
	do
		./ctr --directory $dataname --user $data_path/users_train.dat --item $data_path/items_train.dat --a 1 --b 1 --lambda_u 0.01 --lambda_v 0.01\
		--mult $data_path/mult.dat --theta_init $data_path/theta_2.dat \
		--num_factors $num_factors --max_iter 100 --learning_rate $drate
		echo "Eval $dataname rate $drate $lv"
		python evalb.py -d $dataname -n all -r $dataname-$drate
	done
done
