from pipeline_tool.pipeline_config import PipelineConfig

config = PipelineConfig(input_shape=[1,2], output_shape=[12], data_type="long")

print("A simple config as been created")

config.create_mha_conf_equal(nb_mha=2, num_heads=12, embed_dim=3, dropout=0.0, batch_first=True)

print("An MHA config as been added")
