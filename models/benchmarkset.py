class BenchmarkSet:
    def log(self):
        pass
    def setup(self):
        pass
    def getDataLoader(self):
        pass
    def getAssociatedModel(self):
        pass
    def getAssociatedCriterion(self):
        pass
    def train(self, model, device, train_loader, optimizer, criterion,lr_scheduler,create_graph):
        pass
    def test(self,model, device, test_loader, criterion):
        pass
