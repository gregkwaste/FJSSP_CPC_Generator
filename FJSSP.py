#Main FJSSP Classes
visited_ops = {}

def init_visited_ops(jb):
    global visited_ops
    visited_ops = {}
    #Init visited ops to false
    for o in jb.operations:
        visited_ops[o] = False


class problem:
    def __init__(self):
        self.jobs = []
        self.machines = []
        self.operations = []

    def calc_prob_density(self):
        density = 0
        for jb in self.jobs:
            density += calc_job_density(jb)
        return density

    def load_instance(self, instance,is_cpc=False):
        f = open(instance, 'r')
        lines = f.readlines()
        f.close()

        # Clear problem
        self.jobs = []
        self.machines = []
        opcounter = 0
        mainopcounter = 0

        # Store operations
        jobNum = int(lines[0].split()[0])
        machNum = int(lines[0].split()[1])
        flex = float(lines[0].split()[2])

        #Store machines
        for i in range(machNum):
            self.machines.append(i)

        for i in range(jobNum):
            l = lines[i + 1].strip()
            sp = l.split()

            jb = job()
            jb.uid = i
            job_opnum = int(sp[0])
            l_index = 1

            # Insert Dummy first operation
            op = operation()
            op.dummy = True
            op.label = "J"+str(i)+"S"
            op.uid = opcounter
            op.processing_times = []
            jb.operations.append(op)
            opcounter += 1

            for j in range(job_opnum):
                op = operation()
                op.uid = opcounter
                op.job_ID = i
                op.job_subID = j
                op.old_uid = mainopcounter
                op.label = str(mainopcounter)
                op_opts = int(sp[l_index])
                l_index += 1
                for o in range(op_opts):
                    val = int(sp[l_index + 2 * o + 1])
                    op.processing_times[int(sp[l_index + 2 * o])] = val

                # op.report()
                jb.operations.append(op)
                self.operations.append(op)
                l_index += 2 * op_opts
                opcounter += 1
                mainopcounter += 1

            # Insert Dummy last operation
            op = operation()
            op.dummy = True
            op.label = "J" + str(i) + "E"
            op.uid = opcounter
            op.processing_times = []
            jb.operations.append(op)
            opcounter += 1

            self.jobs.append(jb)

        #Continue parsing CPC instance
        if (is_cpc):
            line_offset = 1 + jobNum
            for i in range(mainopcounter):
                op = self.operations[i];
                # Load predecessors
                sp = lines[line_offset + i].split()
                index = 0;
                pred_count = int(sp[index])
                index += 1

                for j in range(pred_count):
                    op.predecessors.append(self.operations[int(sp[index])])
                    index +=1

                #Load Successors
                suc_count = int(sp[index])
                index += 1

                for j in range(suc_count):
                    op.successors.append(self.operations[int(sp[index])])
                    index +=1


    def export_to_dat(self, output_file):
        f = open(output_file, 'w')
        #Write stats
        f.write("NbJobs = %d;\n" % len(self.jobs))
        f.write("NbMchs = %d;\n" % len(self.machines))
        f.write("NbOps = %d;\n" % len(self.operations))

        #Write Operations
        f.write("Operations = {\n")
        for i in range(len(self.operations)):
            op = self.operations[i]
            f.write("<%d,%d,%d>,\n" % (i, op.job_ID, op.job_subID))
        f.write("};\n")

        #Write Modes
        f.write("Modes = {\n")
        for i in range(len(self.operations)):
            op = self.operations[i]
            for j in op.processing_times.keys():
                mach_id = j
                proc_time = op.processing_times[j]
                f.write("<%d,%d,%d>,\n" % (i, mach_id, proc_time))
        f.write("};\n")

        #Write Processing Times
        f.write("ProcessingTimes = [\n")
        for i in range(len(self.operations)):
            op = self.operations[i]
            op_proc_times = [0] * len(self.machines)
            for j in op.processing_times:
                op_proc_times[j] = op.processing_times[j]
            
            #Write op proc times
            f.write("[" + ",".join(map(str, op_proc_times)) + "],\n")

        f.write("];\n")


        #Write Succession Map
        f.write("SuccessionMap = [\n")
        for i in range(len(self.operations)):
            op = self.operations[i]
            op_suc_map = [0] * len(self.operations)
            for suc in op.successors:
                op_suc_map[suc.old_uid] = 1
            
            #Write op proc times
            f.write("[" + ",".join(map(str,op_suc_map)) + "],\n")

        f.write("];\n")        

        f.close()


    def findPath(self, from_op, to_op):
        init_visited_ops(self)
        from_op.findPathToOp(to_op, 0)


    def findPredPath(self, from_op, to_op):
        init_visited_ops(self)
        from_op.findPredPathToOp(to_op, 0)



class job:
    def __init__(self):
        self.operations = []
        self.uid = None

    def calc_job_density(self):
        '''Calculates Job density as the ratio of the jb nodes
        over the number of arcs in the job'''
        node_num = len(self.operations) - 2
        edge_num = 0

        last_dummy = self.operations[-1]
        for op in self.operations:
            if not op.dummy:
                edge_num += len(op.successors)

            if last_dummy in op.successors:
                edge_num -= 1

        print(node_num, edge_num)
        return float(node_num)/max(1, edge_num)

    def findPath(self, from_op, to_op):
        init_visited_ops(self)
        return from_op.findPathToOp(to_op, 0)
        

    def findPredPath(self, from_op, to_op):
        init_visited_ops(self)
        return from_op.findPredPathToOp(to_op, 0)


class operation:
    def __init__(self):
        self.uid = None #Holds the uid counter including dummy operations
        self.old_uid = None #Holds the real uid counter in case of dummy operations
        self.job_ID = None; #Holds the job id that the operation belongs to
        self.job_subID = None; #Holds the index of the operations within the job operations
        self.label = ""; # Used for plotting
        self.successors = []
        self.predecessors = []
        self.processing_times = {}
        self.dummy = False

    def report(self):
        print("Operation", self.uid)
        print("\tProcessing Times:", self.processing_times)
        print("\tPredecessors: ", [t.uid for t in self.predecessors])

    def findPathToOp(self, op, counter):
        global visited_ops
        visited_ops[self] = True

        if (counter + 1 > 50):
            print("Recursion Limit Reached")
            return False
        
        #Check for paths from the current op to op
        for s in self.successors:
            # print("Visiting Successor", s)
            if (visited_ops[s]):
                continue
            if (s.uid == op.uid):
                return True
            elif (s.findPathToOp(op, counter + 1)):
                return True
        
        return False


    def findPredPathToOp(self, op, counter):
        global visited_ops
        visited_ops[self] = True

        if (counter + 1 > 50):
            print("Recursion Limit Reached")
            return False
        
        #Try to find if any of the predecessors of the current op
        #is an immediate predecessor of op
        for p in self.predecessors:
            if (visited_ops[p]):
                continue

            if p in op.predecessors:
                return True
            elif (p.findPredPathToOp(op, counter + 1)):
                return True
        return False


    def getAllSuccessorsList(self):
        l = self.successors[:]
        for s in self.successors:
            l.extend(s.getAllSuccessorsList())
        return l

