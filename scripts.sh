# CUDA_VISIBLE_DEVICES=2 nohup python main_counterfact_zsre.py --alg_name GRACE --ds_name mcf --model_name EleutherAI/gpt-j-6b  --hparams_fname EleutherAI_gpt-j-6B.json --num_edits 10000 --dataset_size_limit 10000 > logs/grace_gptj_mcf.log &

# CUDA_VISIBLE_DEVICES=1 nohup python main_counterfact_zsre.py --alg_name GRACE --ds_name mcf --model_name Llama-2-7b-hf --hparams_fname Llama-2-7b-hf.json --num_edits 10000 --dataset_size_limit 10000 > logs/grace_llama2_mcf.log &

# CUDA_VISIBLE_DEVICES=2 nohup python main_counterfact_zsre.py --alg_name GRACE --ds_name zsre --model_name EleutherAI/gpt-j-6b --hparams_fname EleutherAI_gpt-j-6B.json --dataset_size_limit 10000 > logs/grace_gptj_zsre.log &

# CUDA_VISIBLE_DEVICES=3 nohup python main_counterfact_zsre.py --alg_name GRACE --ds_name zsre --model_name Llama-2-7b-hf --hparams_fname Llama-2-7b-hf.json --num_edits 10000 --dataset_size_limit 10000 > logs/grace_llama2_zsre.log &


CUDA_VISIBLE_DEVICES=2 nohup python main_ripple_edits.py > logs/grace_gptj_ripple.log &