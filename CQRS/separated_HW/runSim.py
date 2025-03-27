import os, sys
import itertools
import random
import xml.etree.ElementTree as et
from multiprocessing import Pool, Lock
import time
sys.path.insert(1, '../../')
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
	OUTFILE = 'sep_phy.csv'

	N_TOT = 100
	N_r = [90]
	Z_r = [10]
	Z_w = [x/10 for x in range(2000, 10001, 100)]
	C_DB_r = [1]
	S_r = [5]
	C_DB_w = [1]
	S_w = [40]
	
	ATTRIBUTE_NAMES = ['N_read', 'N_write', 'r_Z_read', 'r_Z_write', 'C_DB_read', 'r_DB_read',  'C_DB_write', 'r_DB_write']
	MEASURE_NAMES = ['R0', 'R0_low', 'R0_up', 'R0_read', 'R0_read_low', 'R0_read_up', 'R0_write', 'R0_write_low', 'R0_write_up', 'X0', 'X0_low', 'X0_up', 'X0_read', 'X0_read_low', 'X0_read_up', 'X0_write', 'X0_write_low', 'X0_write_up', 'U_DBread', 'U_DBread_low', 'U_DBread_up', 'U_DBwrite', 'U_DBwrite_low', 'U_DBwrite_up', 'sim_time_sec']
	
	with open(OUTFILE, 'w') as f:
		f.write(','.join(ATTRIBUTE_NAMES) + ',' + ','.join(MEASURE_NAMES) + '\n')
	
	source = ['VAL{:03d}'.format(n) for n in range(len(ATTRIBUTE_NAMES))]
	paramsIterable = []
	for idx, vals in enumerate(list(itertools.product(N_r, Z_r, Z_w, C_DB_r, S_r, C_DB_w, S_w))):
		nr = vals[0]
		nw = N_TOT - nr
		rzr = 1/vals[1]
		rzw = 1/vals[2]
		cr = vals[3]
		rdbr = 1/vals[4]
		cw = vals[5]
		rdbw = 1/vals[6]
		params = [nr, nw, rzr, rzw, cr, rdbr, cw, rdbw]
		paramsIterable.append((idx, source, params, OUTFILE))
	
	l = Lock()	
	with Pool(NUM_SIM_THREADS, initializer=init, initargs=(l,)) as pool:
		pool.starmap(runSim, paramsIterable)	
	pool.close()
	pool.join()
		
		
if __name__ == '__main__':
	main()
