import scarches as sca

condition_key = 'study'
adata = sca.datasets.pancreas()
print(adata.X.min(), adata.X.max())

condition_labels = adata.obs[condition_key].unique().tolist()
network = sca.models.scArches(task_name='test',
                              x_dimension=adata.shape[1],
                              z_dimension=10,
                              device='gpu',
                              loss_fn='nb',
                              architecture=[128, 32],
                              conditions=condition_labels,
                              model_path='./models/')

network.train(adata,
              n_epochs=150,
              condition_key=condition_key,
              steps_per_epoch=100,
              batch_size=128)
# network.save()
#
# new_network = sca.models.scArches.from_config('./models/test/scArchesNB.json')
# new_network.restore_model_config()
# print(new_network.model_path)
