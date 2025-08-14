DATA_path = "/home/WorkSpace/AVQA/Dataset/AVQA-data/"

config = dict(
	type='STG',
	seed=713,
	epochs=20,
	num_labels=42,
	log_interval=100,
	output_dir= './save',
	pretrained_weight="base",
	data=dict(
		root='./data',
		img_size=384,
		batch_size=32,
		eval_batch_size=32,
		num_workers=2,
		frame_sample_rate=1, 

		audios_dir='./raw_audios',
		frames_dir='/home/WorkSpace/AVQA/Dataset/MUSIC-AVQA/frames-1ps',
		train_annot='./annots/music_avqa/music_avqa_train.json',
		valid_annot='./annots/music_avqa/music_avqa_val.json',
		test_annot='./annots/music_avqa/music_avqa_test.json',
        # test_annot='./annots/music_avqa_r/avqa-test-tail.json',  # MUSIC-AVQA-R: [avqa-test-head, avqa-test-tail, avqa-test-headtail]
		test_annots=None,
		ans_quelen='./annots/music_avqa/answer2idx.json',
		
		quest_feat=DATA_path + 'clip_question_L14.h5',
		audio_feat=DATA_path + 'vggish.h5',
		video_feat=DATA_path + 'frame_feat.h5',
		patch_feat=DATA_path + 'tomebl14.h5',
		word_feat=DATA_path + 'clip_word_vit_L14.h5',
		prompt_feat=None,
	),

	hyper_params=dict(
		gpus='0',
		model_type="STG_ViTL14@336px",
		model=dict(
			d_model=512,
			video_dim=768,
			patch_dim=1024,
			quest_dim=768,
			audio_dim=128,
			topK=7,
			num_experts=7,
			encoder_type='ViT-L/14@336px',
			graph_type="MTG",
		),
		optim=dict(
			lr=3e-4,
			encoder_lr=None,
			min_lr=1e-7,
			weight_decay=0.01,
			betas=(0.95, 0.999)
		),
		sched=dict(
			name='StepLR',
			mode='min',
			gamma=0.1,
			decay_start=8,
			decay_every=6,
			factor=0.5,
			patience=5,
			verbose=True,	
			warmup_epochs=2,
		),
	)
)