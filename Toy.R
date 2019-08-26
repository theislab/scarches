library(splatter)
library(scater)

params <- newSplatParams()
params <- setParams(params, update = list(nGenes = 2000))

params <- setParams(params, update = list(batchCells = c(5000, 5000, 5000, 5000)))
# params <- setParams(params, update = list(batch.facLoc = c(0.1))

params <- setParams(params, update = list(mean.shape = 0.6, mean.rate = 0.3))                    
params <- setParams(params, update = list(de.prob = c(0.2)))
# params <- setParams(params, update = list(de.facLoc = c(0.3, 0.3), 
#                                           de.facScale = c(0.1, 0.1)))

params <- setParams(params, update = list(group.prob = c(0.2, 0.2, 0.2, 0.2, 0.2)))

sim <- splatSimulate(params, verbose = FALSE, method = "groups")
sim <- normalize(sim)

plotPCA(sim, colour_by = "Group")

write.table(x = counts(sim), file = "./data/toy/toy.csv", sep = ",")
write.table(x = sim@colData@listData[["Batch"]], file = "./data/toy/toy_batch.csv", sep = ',')
write.table(x = sim@colData@listData[["Group"]], file = "./data/toy/toy_celltype.csv", sep = ',')
