# GOOD dishwasher



# echo dishwasher
# python seq2point_multistate.py --appliance_name dishwasher --batch_size 256 --n_epoch 10 --patience 5 --seed 3409

echo "microwave"
python seq2point.py --appliance_name microwave --batch_size 256 --n_epoch 10 --patience 5 --seed 3409

# echo "fridge"
# python seq2point_multistate.py --appliance_name fridge --batch_size 256 --n_epoch 10 --patience 5 --seed 779

# echo "kettle"
# python seq2point_multistate.py --appliance_name kettle --batch_size 256 --n_epoch 10 --patience 5 --seed 779

# echo "washingmachine"
# python seq2point_multistate.py --appliance_name washingmachine --batch_size 256 --n_epoch 10 --patience 5 --seed 3407


# echo "microwave"
# python my_s2s.py --appliance_name microwave --batch_size 256 --n_epoch 10 --patience 5 --seed 3409
# echo "fridge"
# python my_s2s.py --appliance_name fridge --batch_size 256 --n_epoch 10 --patience 5 --seed 3409
# echo "washingmachine"
# python my_s2s.py --appliance_name washingmachine --batch_size 256 --n_epoch 10 --patience 5 --seed 3409
# echo "dishwasher"
# python my_s2s.py --appliance_name dishwasher --batch_size 256 --n_epoch 10 --patience 5 --seed 3409