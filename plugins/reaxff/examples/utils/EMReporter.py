from openmm import MinimizationReporter

class EnergyMinimizationReporter(MinimizationReporter):
    interval = 10 # report interval
    energies = [] # array to record progress
    def __init__(self, output_file):
        super(EnergyMinimizationReporter, self).__init__()
        self.output_file = output_file
    def report(self, iteration, x, grad, args):
        # print current system energy to screen 
        if iteration % self.interval == 0:
            print(iteration, args['system energy'])
            with open(self.output_file, 'a') as file:  # Append mode
                file.write(f"{iteration},{args['system energy']}\n")
            
        # save energy at each iteration to an array we can use later
        self.energies.append(args['system energy'])
        
        

        # The report method must return a bool specifying if minimization should be stopped. 
        # You can use this functionality for early termination.
        return False
