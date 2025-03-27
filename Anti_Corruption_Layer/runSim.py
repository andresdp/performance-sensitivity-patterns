import os, sys
import itertools
import random
import xml.etree.ElementTree as et
from multiprocessing import Pool, Lock
import time
sys.path.insert(1, '../')
from varEnv import * # To import *jmtpath* and *NUM_SIM_THREADS*



def genModel(idx, source, params, modelName):
	placeholderName = 'model.placeholder.jsimg'
	with open(placeholderName, 'r') as f:
		filedata = f.read()
	for s, p in zip(source, params):
		filedata = filedata.replace(s, str(p))
	filedata = filedata.replace(placeholderName, modelName)
	with open(modelName, 'w') as f:
		f.write(filedata)
			
			
def runSim(idx, source, params, outfile):
	modelName = 'model' + str(idx) + '.jsimg'
	resultName = modelName + '-result.jsim'
	genModel(idx, source, params, modelName)
	rnd = str(random.randint(0, sys.maxsize))
	cmd = 'java -cp ' + jmtpath + ' jmt.commandline.Jmt sim ' + modelName + ' -seed ' + rnd
	start_time = time.time()
	os.popen(cmd).read()
	end_time = time.time()
	collectResults(params, resultName, end_time - start_time, outfile)
	os.popen('rm ' + modelName + ' ' + resultName)
		
		
def collectResults(params, resultName, sim_time, outfile):
	tree = et.parse(resultName)
	root = tree.getroot()
	strAttrVals = ','.join([str(p) for p in params])
	for measure in root:
		dictAttrib = measure.attrib
		strAttrVals = strAttrVals + ',' + ','.join([dictAttrib['meanValue'], dictAttrib['lowerLimit'], dictAttrib['upperLimit']])
	strAttrVals = strAttrVals + ',' + str(sim_time)
	with lockWrite:
		with open(outfile, 'a') as f:
			f.write(strAttrVals + '\n')
	
		
def init(l):
    global lockWrite
    lockWrite = l		
	
	
def main():
	OUTFILE = 'multi_N25.csv'

	N_TOT = 25
	N_A = range(0, N_TOT+1, 1)
	Z_A = [20]
	Z_B = [100]
	C_SS1 = [1]
	p_A_SS1toACL = [0.2]
	p_B_SS1toACL = [0.8]
	S_A_SS1 = [12]
	S_B_SS1 = [0.6]
	S_A_ACL = [20]
	S_B_ACL = [5]
	S_A_SS2 = [32]
	S_B_SS2 = [4.25]
	
	ATTRIBUTE_NAMES = ['N_A', 'N_B', 'r_Z_A', 'r_Z_B', 'C_SS1', 'r_A_SS1', 'r_B_SS1', 'p_SS1toACL_A', 'p_SS1toRef_A', 'p_SS1toACL_B', 'p_SS1toRef_B', 'r_A_ACL', 'r_B_ACL', 'r_A_SS2', 'r_B_SS2']
	MEASURE_NAMES = ['R0', 'R0_low', 'R0_up', 'R0_A', 'R0_A_low', 'R0_A_up', 'R0_B', 'R0_B_low', 'R0_B_up', 'X0', 'X0_low', 'X0_up', 'X0_A', 'X0_A_low', 'X0_A_up', 'X0_B', 'X0_B_low', 'X0_B_up', 'Uss1', 'Uss1_low', 'Uss1_up', 'Uacl', 'Uacl_low', 'Uacl_up', 'Uss2', 'Uss2_low', 'Uss2_up', 'sim_time_sec']
	
	with open(OUTFILE, 'w') as f:
		f.write(','.join(ATTRIBUTE_NAMES) + ',' + ','.join(MEASURE_NAMES) + '\n')
	
	source = ['VAL{:03d}'.format(n) for n in range(len(ATTRIBUTE_NAMES))]
	paramsIterable = []
	for idx, vals in enumerate(list(itertools.product(N_A, Z_A, Z_B, C_SS1, p_A_SS1toACL, p_B_SS1toACL, S_A_SS1, S_B_SS1, S_A_ACL, S_B_ACL, S_A_SS2, S_B_SS2))):
		nA = vals[0]
		nB = N_TOT - nA
		rzA = 1/vals[1]
		rzB = 1/vals[2]
		css1 = vals[3]
		pAtoAcl = vals[4]
		pAtoRef = 1.0 - pAtoAcl
		pBtoAcl = vals[5]
		pBtoRef = 1.0 - pBtoAcl
		rAss1 = 1/vals[6]
		rBss1 = 1/vals[7]
		rAacl = 1/vals[8]
		rBacl = 1/vals[9]
		rAss2 = 1/vals[10]
		rBss2 = 1/vals[11]
		params = [nA, nB, rzA, rzB, css1, rAss1, rBss1, pAtoAcl, pAtoRef, pBtoAcl, pBtoRef, rAacl, rBacl, rAss2, rBss2]
		paramsIterable.append((idx, source, params, OUTFILE))
	
	l = Lock()	
	with Pool(NUM_SIM_THREADS, initializer=init, initargs=(l,)) as pool:
		pool.starmap(runSim, paramsIterable)	
	pool.close()
	pool.join()
		
		
if __name__ == '__main__':
	main()
