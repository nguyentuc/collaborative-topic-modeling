#! /bin/bash
dataname=data_lazada
data_path="../../Data/$dataname"

num_factors=100

#root_path=../data/arxiv/cv
#num_factors=100
#./ctr --directory $dataname/ --user $data_path/cf-train-users.dat --item \
#  $data_path/cf-train-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
#  --mult $data_path/mult.dat --theta_init $data_path/theta_2.dat \
#  --beta_init $data_path/beta_final.dat --num_factors $num_factors --save_lag 20

 ./ctr --directory $dataname/ --user $data_path/users_train.dat --item \
  $data_path/items_train.dat --a 1 --b 1 --lambda_u 0.01 --lambda_v 0.01 \
  --mult $data_path/mult.dat --theta_init $data_path/theta_2.dat \
  --beta_init $data_path/beta_final.dat --num_factors $num_factors --save_lag 20 