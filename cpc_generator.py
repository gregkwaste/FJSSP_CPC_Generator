# Imports
from random import randint, seed, sample, uniform
from math import ceil
from copy import deepcopy
import graphviz as pgv
import sys
import os
from FJSSP import problem, job, operation


seed_value = 111111
MAX_INT = 2^32
#seed_value = randint(-MAX_INT, MAX_INT)
seed(seed_value)
# Main classes

#global variables
visited_ops = {}

def params_preprocessor(params):
    #Make sure that the generation parameters are valid
    params["max_jobs"] = max(params["min_jobs"], params["max_jobs"])
    params["max_machs"] = max(params["min_machs"], params["max_machs"])
    params["max_preds"] = max(params["min_preds"], params["max_preds"])
    params["max_sucs"] = max(params["min_sucs"], params["max_sucs"])
    params["max_flex"] = max(params["min_flex"], params["max_flex"])
    #Make sure that there cannot be more available machines than the max machine num
    params["max_flex"] = min(params["max_machs"], params["max_flex"])


def remove_edges(jb, params):
    possible_edges = []
    for target_op in jb.operations:
        if (target_op.dummy):
            continue

        if (len(target_op.predecessors) == 1):
            continue

        for source_op in jb.operations:
            if source_op.dummy:
                continue
            if not source_op in target_op.predecessors:
                continue
            if (len(source_op.successors) == 1):
                continue

            #It should be possible to be able to remove this edge
            possible_edges.append((source_op, target_op))
        

    if not len(possible_edges):
        print("No Move Found")
        return 1000000

    #sort the edges by the uid difference
    #possible_edges.sort(key=lambda x: x[1].uid - x[0].uid)
    
    #Select the edge with the closest operations
    edge = possible_edges[randint(0, len(possible_edges)-1)]
    #edge = possible_edges[0]
    print ([op.uid for op in edge[0].successors])
    print ([op.uid for op in edge[1].predecessors])
    print("Removing Edge between", edge[0].uid, edge[1].uid)
    edge[0].successors.remove(edge[1])
    edge[1].predecessors.remove(edge[0])

    print ([op.uid for op in edge[0].successors])
    print ([op.uid for op in edge[1].predecessors])

    return jb.calc_job_density()


def add_edges(jb, params):
    possible_edges = []
    for target_op in jb.operations:
        if (target_op.dummy):
            continue

        if (len(target_op.predecessors) == params["max_preds"]):
            continue

        for source_op in jb.operations:
            if source_op.dummy:
                continue
            if (source_op.uid >= target_op.uid):
                continue
            if (len(source_op.successors) == params["max_sucs"]):
                continue

            #Check if there is already a path from source to target op
            if (jb.findPath(source_op, target_op)):
                continue

            if (jb.findPredPath(source_op, target_op)):
                continue
            
            #I should be able to add an edge between them at this point
            possible_edges.append((source_op, target_op))
        

    if not len(possible_edges):
        print("No Move Found")
        return 1000000

    #sort the edges by the uid difference
    #possible_edges.sort(key=lambda x: x[1].uid - x[0].uid)
    
    #Select the edge with the closest operations
    edge = possible_edges[randint(0, len(possible_edges)-1)]
    #edge = possible_edges[0]
    print("Adding Edge between", edge[0].uid, edge[1].uid)
    edge[0].successors.append(edge[1])
    edge[1].predecessors.append(edge[0])

    return jb.calc_job_density()



def fix_isolated_ops(params, jb):
    #Look for isolated operations
    for op in jb.operations:
        cond_1 = len(op.successors)==0 #Operation has no successors
        cond_2 = len(op.predecessors)==1 #Operation has one predecessor
        cond_3 = jb.operations[0] in op.predecessors #The operations predecessor is 0

        if cond_1 and cond_2 and cond_3:
            print("Fixing isolated operation", op.uid)
            cands = []
            #Find new predecessor candidates
            for cand in jb.operations:
                if (cand.uid >= op.uid):
                    continue

                if (len(cand.successors) == params['max_sucs']):
                    continue
                else:
                    cands.append(cand)

            #Select one candidate at random
            cand = cands[randint(0, len(cands)-1)]

            #Do the connections
            cand.successors.append(op)
            op.predecessors.append(cand)

            #Check if path from the first dummy to cand exists
            #if so remove it from op's predecessors
            if (jb.findPath(jb.operations[0], cand)):
                op.predecessors.remove(jb.operations[0])
                jb.operations[0].successors.remove(op)


def converge_with_LS(params, jb, target_density):
    iters = 20
    epsilon = 0.01
    new_jb = deepcopy(jb)


    current_cost = jb.calc_job_density()
    absdiff = abs(current_cost - target_density)
    for i in range(iters):
        #Select mode: 0 if we need to lower the density/ add edges
        #Select mode: 1 if we need to increase the density/ remove edges
        if (current_cost < target_density):
            mode = 1
        else:
            mode = 0

        #Select an operation randomly except of the dummies
        op =  jb.operations[randint(1, len(jb.operations)-1)]

        temp_jb = deepcopy(new_jb)

        new_cost = 10000000
        if (mode == 0): #Add Edges
            new_cost = add_edges(temp_jb, params)
        elif (mode ==1):
            new_cost = remove_edges(temp_jb, params)

        new_absdiff = abs(new_cost - target_density)

        #Check for better solution
        if (mode ==0):
            if ((new_cost < current_cost) and (new_absdiff < absdiff)):
                print("Better Cost", new_cost)
                new_jb = temp_jb
                current_cost = new_cost
                absdiff = new_absdiff
            else:
                print("Unable to find a better move")
                break
        elif (mode ==1):
            if ((new_cost > current_cost) and (new_absdiff < absdiff)):
                print("Better Cost", new_cost)
                new_jb = temp_jb
                current_cost = new_cost
                absdiff = new_absdiff
            else:
                print("Unable to find a better move")
                break

        #Exit condition
        if (abs(target_density - current_cost) < epsilon):
            break

    return new_jb


def generator(prob, params, plot_flag):
    # Generate trees
    preds = []
    target_density = uniform(params["min_density"], params["max_density"])
    global visited_ops

    for jb_id, jb in enumerate(prob.jobs):
        print(len(jb.operations))
        preds = [jb.operations[0]]
        opcounter = 1
        edge_counter = 0

        while (opcounter < len(jb.operations) - 1):
            jb_density = jb.calc_job_density()
            print("Current Density", jb_density, "Target Density", target_density)
            distance_diff = target_density - jb_density
            remaining_steps = len(jb.operations) - 1 - opcounter

            edges_to_target = int((len(jb.operations) - 2) / target_density) - max(1, edge_counter)

            #Calculate edges per step to catch target
            edges_per_step =  int(ceil(float(edges_to_target) / remaining_steps))
            print("Remaining Edges to reach the target", edges_to_target, 
                  "Remaining Steps to complete the job", remaining_steps, 
                  "Avg Edges Per Step", edges_per_step)

            edges_per_step = max(1, edges_per_step)

            
            pred_limit = edges_per_step + (-1)**randint(1, 2)

            #Clamp to fit the input parameters
            pred_limit = min(max(pred_limit, params["min_preds"]), params["max_preds"])

            op = jb.operations[opcounter]
            print("Working on operation", op.uid, op.old_uid, op)
            
            n = pred_limit
            print("Number of Predecessors to choose", n)
            # Select unique predecessors for op
            inserted_preds = 0
            failCounter = 0
            failLimit = 20
            avail_preds = list(preds)

            while(inserted_preds < n):
                #Select a candidate predecessor at random
                cand_pred = randint(0, len(avail_preds) - 1)
                cand_op = avail_preds[cand_pred]
                #Check if a path already exists from cand_pred to op
                
                insertion_fail = False
                if (cand_op in op.predecessors):
                    insertion_fail = True
                if (len(cand_op.successors) == params["max_sucs"]):
                    insertion_fail = True
                
                if (jb.findPath(cand_op, op)):
                    print("Excluding", cand_op.uid, "Found Path from ", cand_op.uid, "to", op.uid)
                    insertion_fail = True
                
                if (jb.findPredPath(cand_op, op)):
                    print("Excluding", cand_op.uid, "Found Path from a Pred of", cand_op.uid, "to", op.uid)
                    insertion_fail = True
                
                if (insertion_fail):
                    failCounter += 1
                else:
                    print("Inserted Predecessor", cand_op.uid, cand_op)
                    op.predecessors.append(cand_op)
                    cand_op.successors.append(op)
                    inserted_preds += 1

                avail_preds.remove(cand_op)

                if not len(avail_preds):
                    break

            print("Finally Inserted Predecessors", inserted_preds)

            # Add op to preds
            preds.append(op)

            # Filter preds if they have reached a max outdegree
            for i in range(len(preds) - 1, -1, -1):
                p = preds[i]
                if (len(p.successors) == params["max_sucs"]):
                    print("Removing", p.uid, "from set")
                    del preds[i]

            opcounter += 1
            edge_counter += n

        #Fix isolated operations
        fix_isolated_ops(params, jb)

        print("Pre-LS Density", jb.calc_job_density())

        #Try to repair instance with LS
        jb = converge_with_LS(params, jb, target_density)


        # Connect last dummy op to all open operations
        last_op = jb.operations[-1]
        for op in jb.operations[:-1]:
            if not len(op.successors):
                last_op.predecessors.append(op)
                op.successors.append(last_op)

        print("Final Job Density:", jb.calc_job_density())
        #Save job
        prob.jobs[jb_id] = jb

    
    output_file = "CPC_%d_%d_%d_%5.4f.fjs" % (len(prob.jobs),
        len(prob.machines), 
        sum([len(prob.jobs[i].operations) - 2 for j in prob.jobs]),
        target_density)

    #Return the file name
    return output_file
        

def save(prob, save_file):
    print("saving to dataset", save_file)
    f = open(save_file, 'w')
    f.write(" ".join(list(map(str, [len(prob.jobs), len(prob.machines), 1, 1, seed_value]))) + "\n")
    for j in range(len(prob.jobs)):
        jb = prob.jobs[j]
        # f.write(str(len(jb.operations) - 2) + " ") # used for dummies
        f.write(str(len(jb.operations) - 2) + " ")
        for op in jb.operations[:]:
            if (op.dummy):
                continue
            # Write processing times
            f.write(str(len(op.processing_times)) + " ")
            for p, val in op.processing_times.items():
                # Use zero indexing for the machines as well
                f.write(" ".join(list(map(str, (p - 1, val)))) + " ")
        f.write('\n')

    # Write successors once all operations have been defined
    for j in range(len(prob.jobs)):
        jb = prob.jobs[j]
        for op in jb.operations[:]:
            if (op.dummy):
                continue

            # Filter predecessors
            true_preds = op.predecessors[:]
            for i in range(len(true_preds) - 1, -1, -1):
                s = true_preds[i]
                if s.dummy:
                    del true_preds[i]

            # Filter successors
            true_sucs = op.successors[:]
            for i in range(len(true_sucs) - 1, -1, -1):
                s = true_sucs[i]
                if s.dummy:
                    del true_sucs[i]

            # Write predecessors
            f.write(str(len(true_preds)) + " ")
            for s in true_preds:
                f.write(str(s.old_uid) + " ")

            # Write successors
            f.write(str(len(true_sucs)) + " ")
            for s in true_sucs:
                f.write(str(s.old_uid) + " ")
            f.write('\n')

    f.close()


def plot(prob, plot_name):
    print("Plotting")
    # Plot Network
    G = pgv.Digraph(format='png')
    # set defaults
    G.node_attr = {'shape': 'circle'}
    G.graph_attr = {'rankdir': 'LR', 'ordering': 'out',
                    'nodesep': '0.5', 'ranksep': '1', 'size': "30, 60"}
    G.edge_attr = {'arrowsize': '1.0'}

    for j in range(len(prob.jobs) - 1, -1, -1):
        jb = prob.jobs[j]
        with G.subgraph(name='cluster' + str(jb.uid)) as c:
            #c.attr(style='invis')
            c.attr(label='Density ' + str(jb.calc_job_density()))
            c.attr(fontsize='20')
            c.attr(color='invis')
            for i in range(len(jb.operations)):
                op = jb.operations[i]
                G.node(op.label, **{'width':"0.5", 'height':"0.5"})
                # Add Edge
                for s in op.successors:
                    c.edge(op.label, s.label)

    # print(G.source)
    G.render(plot_name)
    # G.view('test')

    # A = to_agraph(G)
    # A.layout('dot')
    # print(A)
    # A.draw('test.png')
    return G



def instance_generator_usage():
    print("Options:")
    print('min_jobs','max_jobs','min_machs','max_machs','min_proc_time',
        'max_proc_time', 'min_preds', 'max_preds','min_sucs','max_sucs',
        'min_ops','max_ops','min_flex','max_flex','min_density','max_density')


def generate_problem(params):
    #Generate problem 
    prob = problem()

    #Select number of jobs
    job_num = randint(params["min_jobs"], params["max_jobs"])
    #Select number of machines
    mach_num = randint(params["min_machs"], params["max_machs"])
    print("Job Num", job_num, "Machine Num", mach_num)

    for i in range(mach_num):
        prob.machines.append(i + 1)

    #Generate Jobs
    opcounter = 0
    mainopcounter = 0
    for i in range(job_num):
        jb = job()
        jb.uid = i

        #Select number of operations for the job
        op_num = randint(params["min_ops"], params["max_ops"])
        #Populate jobs with operations
        # Insert Dummy first operation
        op = operation()
        op.uid = opcounter
        op.dummy = True
        jb.operations.append(op)
        opcounter += 1
        for j in range(op_num):
            op = operation()
            op.uid = opcounter
            op.old_uid = mainopcounter
            #Select number of available machines for the operation
            opt_num = randint(params["min_flex"], params["max_flex"])
            opts = sample(range(mach_num), opt_num)
            
            #print("Operation", op.uid)
            #Select machines for the operation
            for o in range(opt_num):
                val = randint(params["min_proc_time"], params["max_proc_time"])
                op.processing_times[opts[o] + 1] = val #Use 1 index for machines

            #print(op.processing_times)

            # op.report()
            jb.operations.append(op)
            opcounter += 1
            mainopcounter += 1

        # Insert Dummy last operation
        op = operation()
        op.uid = opcounter
        op.dummy = True
        jb.operations.append(op)
        opcounter += 1

        prob.jobs.append(jb)

    return prob


def instance_generator(params):
    #Generate problem
    prob = generate_problem(params)

    #Call generator
    output_file = generator(prob, params, True)

    return (output_file, prob)


def convert_fjssp_to_fjsspcpc(instance, params, prefix, plot_flag):
    #Generate problem
    prob = problem()

    #load instance
    prob.load_instance(instance);

    #Generate Graph for the instance
    output_file = generator(prob, params, plot_flag)

    return ("_".join([prefix, output_file]), prob)



def main(sysargs):

    if (len(sysargs) < 2):
        print("Missing Arguments - Default Mode- Converting a hardcoded fjs instance")
        # sample_instance = '/media/DATA/greg_data/Source/Repos/SchedulingCode/Benchmarks/FJSSPinstances/6_Fattahi/Fattahi20.fjs'
        # sample_instance = 'Benchmarks/FJSSPinstances/1_Brandimarte/BrandimarteMk3.fjs'
        sample_instance = 'BrandimarteMk10.fjs'
        # sample_instance = 'Benchmarks/FJSSPinstances/6_Fattahi/Fattahi19.fjs'
        # sample_instance = 'Benchmarks\\FJSSPinstances\\1_Brandimarte\\BrandimarteMk3.fjs'

        params = {'min_jobs': -1,
                  'max_jobs': -1,
                  'min_machs': -1,
                  'max_machs': -1,
                  'min_proc_time': -1,
                  'max_proc_time': -1,
                  'min_preds': 3,
                  'max_preds': 3,
                  'min_sucs': 3,
                  'max_sucs': 3,
                  'min_ops': -1,
                  'max_ops': -1,
                  'min_flex': 4,
                  'max_flex': 4,
                  'min_density': 0.5,
                  'max_density': 0.7}

        #Generate new output name
        sp = sample_instance.split('/')
        prefix = sp[-1].split('.')[0] 
        
        #Load Instance
        output_file, prob = convert_fjssp_to_fjsspcpc(sample_instance, params, prefix, True)

        #Save instance
        save(prob, output_file)

        #Plot Instance
        plot(prob, ".".join(output_file.split(".")[0:-1]))
        return prob
 
    else:
        #Try to parse options
        for arg in sysargs:
            if (arg == '-A'):
                #Convert all instances
                for subdir, dirs, files in os.walk("Benchmarks/FJSSPinstances"):
                    for file in files:
                        #print os.path.join(subdir, file)
                        filepath = subdir + os.sep + file
                        if ('Arcelik' in filepath or 'arcelik' in filepath):
                            continue
                        if ('lines' in filepath):
                            continue
                        if ('cpc' in filepath):
                            continue
                        if not filepath.endswith('.fjs'):
                            continue
                        #New filepath
                        subdir_split = subdir.split(os.sep)
                        outputfilepath = subdir_split[0] + os.sep + 'CPC_' + os.sep.join(subdir_split[1:]) + os.sep + file
                        
                        if not os.path.exists(os.path.dirname(outputfilepath)):
                            os.makedirs(os.path.dirname(outputfilepath))
                        
                        try:
                            generator(filepath, outputfilepath, 1)
                            print('Saved to file: ', outputfilepath)
                        except:
                            print('Paixtike malakia')
                            print(filepath)
                exit = True
            elif (arg == '-G'):
                print("Generating new instance")
                #Parse required dataset attributes
                args = sysargs[2:]
                if (len(args) < 14):
                    print("Missing Necessary generation parameters")
                    instance_generator_usage()
                    break
                
                params = {'min_jobs': int(args[0]),
                          'max_jobs': int(args[1]),
                          'min_machs': int(args[2]),
                          'max_machs': int(args[3]),
                          'min_proc_time': int(args[4]),
                          'max_proc_time': int(args[5]),
                          'min_preds': int(args[6]),
                          'max_preds': int(args[7]),
                          'min_sucs': int(args[8]),
                          'max_sucs': int(args[9]),
                          'min_ops': int(args[10]),
                          'max_ops': int(args[11]),
                          'min_flex': int(args[12]),
                          'max_flex': int(args[13]),
                          'min_density': float(args[14]),
                          'max_density': float(args[15])}
                
                print("Generating new instance with parameters: ", params)

                #Preprocess the parameters
                params_preprocessor(params)

                #Call generator
                output_file, prob = instance_generator(params)

                #Save instance
                save(prob, output_file)
                


                exit = True
        return prob
    
    
    # If nothing exceptional happened and the app is not exiting, continue
        if not exit:
            #Convert input file
            generator(sys.argv[1], sys.argv[2], sys.argv[3])


if ( __name__ == '__main__' ):
    main(sys.argv)
    


