
class LoadResults:
    def __init__(self,):
        self.all_results = []
        self.all_host_data = []
        self.all_accelerator_data = []
    
    def add_data(self, results, host_data, acc_data):
        self.all_results.append(results)
        self.all_host_data.append(host_data)
        self.all_accelerator_data.append(acc_data)

    def get_all(self):
        return self.all_results, self.all_host_data, self.all_accelerator_data


