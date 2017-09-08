'''
Created on Sep 07, 2017

@author: Varela

motivation: parser for sentiment analysis sentences

'''

import re 

class Tree(object):
	def __init__(self):
		self.left=None 
		self.right=None 
		self.word=None 


def treeParser(text):
	l0, l1, r0, r1 = parse(text)
	node = Tree() 
	if r0 is None and r1 is None: 				
		node.word= text[l0:l1+1]
	else: 
		node.left = treeParser(text[l0:l1+1]) 	
		node.right= treeParser(text[r0:r1+1]) 	

	return node 

def parse(string):
	'''
		Returns l1,l2 
	'''
	L, R= [None, None], [None, None]

	#Invalid string 
	descent= re.compile('\(\d (.*)\)')
	result=  descent.match(string)
	if result is None: 
		return L[0], L[1], R[0], R[1] 
	
	#Root node is word
	substring = result.group(1)
	root= re.compile('^[a-zA-Z]')
	result= root.match(substring)
	if result:
		L[0]=3
		L[1]=len(substring)+2 
		return L[0], L[1], R[0], R[1] 

	L[0]=0
	L[1]=1 
	R[0]=len(substring)-2
	R[1]=len(substring)-1

	if substring[0] ==  '(':  
		LEFT_STOP_TOKEN = ')' 
		Lheight = 1 
	else: 
		LEFT_STOP_TOKEN =' '
		Lheight = 0 
		
	if substring[-1] == ')': 
		RIGHT_STOP_TOKEN = '(' 
		Rheight = 1 
	else: 
		RIGHT_STOP_TOKEN = ' '		
		Rheight = 0 
	
	LEFT_HEIGHT_TOKEN = '(' 
	RIGHT_HEIGHT_TOKEN = ')' 

	
	
	found_left= False 
	found_right= False 	

	for i in xrange(len(substring)):
		# print 'L1:', L[1], 'height', Lheight, 'found', found_left, substring[L[1]]
		# print 'R0:', R[0], 'height', Rheight,  substring[R[0]]
		if substring[L[1]] ==  LEFT_HEIGHT_TOKEN: 
			Lheight +=1 
		elif substring[L[1]] ==  LEFT_STOP_TOKEN: 
			Lheight -=1 

		if substring[R[0]] ==  RIGHT_HEIGHT_TOKEN: 
			Rheight +=1 
		elif substring[R[0]] ==  RIGHT_STOP_TOKEN: 
			Rheight -=1 

		if not found_left:  	
			found_left  = (Lheight == 0) and (substring[L[1]] ==  LEFT_STOP_TOKEN) 	
		if not found_right:  	
			found_right = (Rheight == 0) and (substring[R[0]] ==  RIGHT_STOP_TOKEN) 	
		if found_left and found_right:
			break			
		if not found_left: 
			L[1] +=1
		if not found_right: 
			R[0] -=1

	#ADJUST FOR REGEX MATCHER
	L[0]+=3
	L[1]+=3
	R[0]+=3
	R[1]+=3
	return L[0],L[1],R[0],R[1]



def main():
	# phrase1 = '(5 (5 Great) (3 movie))'	
	# l0, l1, r0, r1 = parse(phrase1)
	# print phrase1[l0:l1+1], phrase1[r0:r1+1]

	# phrase2 = '(5 Great)'	
	# l0, l1, r0, r1 = parse(phrase2)
	# print phrase2[l0:l1+1], r0, r1

	# print 'tree 1', phrase1[l0:l1+1]
	# l0, l1, r0, r1 = parse(phrase1[l0:l1+1])
	# print 'subtree 1.1', phrase1[l0:l1+1]
	# print 'subtree 1.2', phrase1[r0:r1+1]
	# l0, l1, r0, r1 = parse(phrase1[r0:r1+1])
	# print 'tree 2', phrase1[r0:r1+1]
	# print 'subtree 2.1', phrase1[l0:l1+1]
	# print 'subtree 2.2', phrase1[r0:r1+1]

	phrase3= '(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))'
	l0, l1, r0, r1 = parse(phrase3)
	phrase31, phrase32= phrase3[l0:l1+1], phrase3[r0:r1+1]  
	print 'tree 1', phrase31
	print 'tree 2', phrase32

	
	l0, l1, r0, r1 = parse(phrase31)
	print 'tree 1.1', phrase31[l0:l1+1]
	print 'tree 1.2', phrase31[r0:r1+1]
	
	l0, l1, r0, r1 = parse(phrase32)
	print 'tree 2.1', phrase32[l0:l1+1]
	print 'tree 2.2', phrase32[r0:r1+1]
	print phrase32
	# l0, l1, r0, r1 = parse(phrase32)
	# print l0, l1, r0, r1
	# print 'tree 2.1', phrase32[l0:l1+1]
	# print 'tree 2.2', phrase32[r0:r1+1]

	tree = treeParser('(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))')	
	print tree 
	print  'node 1.word',  tree.word  
	print  'node 1.left',  tree.left 
	print  'node 1.right', tree.right 

	print  'node 2.left.word',  tree.left.word  
	print  'node 2.left.left',  tree.left.left 
	print  'node 2.left.right', tree.left.right 

	print  'node 3.left.word',  tree.right.word  
	print  'node 3.left.left',  tree.right.left 
	print  'node 3.left.right', tree.right.right 

	print  'node 4.left.left.word',   tree.left.left.word  
	print  'node 4.left.left.left',   tree.left.left.left 
	print  'node 4.left.left.right',  tree.left.left.right 

	print  'node 5.left.right.word',  tree.left.right.word  
	print  'node 5.left.right.left',  tree.left.right.left 
	print  'node 5.left.right.right', tree.left.right.right 

	print  'node 6.right.left.word',   tree.right.left.word  
	print  'node 6.right.left.left',   tree.right.left.left 
	print  'node 6.right.left.right',  tree.right.left.right 

	print  'node 7.right.right.word',  tree.right.right.word  
	print  'node 7.right.right.left',  tree.right.right.left 
	print  'node 7.right.right.right', tree.right.right.right 




if __name__ == '__main__': 
	main()	




