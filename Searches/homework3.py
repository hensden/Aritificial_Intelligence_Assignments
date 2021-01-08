import heapq as pq
from math import sqrt


class Search():
	def __init__(self, mode, X, Y, Z, entX, entY, entZ, exX, exY, exZ, N, moves):
		self.action = [[], [1,0,0],[-1,0,0],[0,1,0],[0,-1,0], [0,0,1], [0,0,-1],
		[1,1,0], [1,-1,0],[-1,1,0],[-1,-1,0],
		[1,0,1], [1,0,-1],[-1,0,1],[-1,0,-1],
		[0,1,1], [0,1,-1],[0,-1,1],[0,-1,-1]]
		self.mode = mode
		self.X, self.Y, self.Z = X, Y, Z
		self.x1, self.y1, self.z1 = entX, entY, entZ
		self.x2, self.y2, self.z2 = exX, exY, exZ
		self.num = N
		self.tmp = moves
		self.moves = {}
		# op variables
		self.reached = False
		self.score = 0
		self.n = 0
		self.path = []

	def parse(self):
		for item in self.tmp:
			elem = item.split()
			x, y, z = int(elem[0]), int(elem[1]), int(elem[2])
			if x not in self.moves:
				self.moves[x] = {}
			if y not in self.moves[x]:
				self.moves[x][y] = {}
			if z not in self.moves[x][y]:
				self.moves[x][y][z] = []
			for i in elem[3:]:
				self.moves[x][y][z].append(int(i))

	def inrange(self, tup):
		if tup[0] < 0 or tup[0] >= self.X or tup[1] < 0 or tup[1] >= self.Y or tup[2] < 0 or tup[2] >= self.Z:
			return False
		return True


	def bfs(self):
		parent = {}
		Q = [(self.x1, self.y1, self.z1)]
		while len(Q) > 0 and not self.reached:
			L = len(Q)
			for i in range(L):
				# front
				F = Q.pop(0)
				if F[0] not in self.moves or F[1] not in self.moves[F[0]] or F[2] not in self.moves[F[0]][F[1]]:
					continue
				for idx in self.moves[F[0]][F[1]][F[2]]:
					disp = self.action[idx]
					neighbour = (F[0]+disp[0], F[1]+disp[1], F[2]+disp[2])
					if neighbour[0] == self.x2 and neighbour[1] == self.y2 and neighbour[2] == self.z2:
						self.reached = True
					if not self.inrange(neighbour) or str(neighbour)[1:-1] in parent:
						continue
					# parent of neighbour is curr
					parent[str(neighbour)[1:-1]] = F
					Q.append(neighbour)
			self.n += 1
		self.score = self.n
		self.n += 1
		node = (self.x2, self.y2, self.z2)
		if not self.reached:
			return 
		while node != (self.x1, self.y1, self.z1):
			self.path.append(''.join(str(node)[1:-1].split(',')) + ' 1')
			node = parent[str(node)[1:-1]]
		self.path.append(''.join(str(node)[1:-1].split(',')) + ' 0')

	def get_s(self, child, parent):
		d = 0
		for i in range(3):
			d += abs(child[i] - parent[i])
		if d>1:
			return '14'
		else:
			return '10'

	def beautify(self, F):
		return tuple([int(coord.strip()) for coord in F.split('x')[0][1:-1].split(',')])

	def ucs(self):
		# initialize priority queue, visited set and parent map
		parent = {}
		counter = {}
		visited = set([])
		root = (self.x1, self.y1, self.z1)
		counter[str(root)] = 1
		Q = [(0, str(root) + 'x' + str(counter[str(root)]))]
		F, Fnode = None, None
		# F : '(x, y, z)x<count>' and Fnode is (x,y,z) 
		while not self.reached and len(Q) > 0:
			# Pop node with minimum cost (first node in front of queue)
			front = Q.pop(0)
			cost = front[0]
			F = front[1]
			# If node has already been expanded, continue
			# Get numeric representation of node
			Fnode = self.beautify(F)
			if str(Fnode) in visited:
				continue;
			# Check if node==goal
			if Fnode[0]==self.x2 and Fnode[1]==self.y2 and Fnode[2]==self.z2:
				self.score = cost
				self.reached = True
			# Mark node as expanded/visited
			visited.add(str(Fnode))
			# Break if goal is reached
			if self.reached:
				continue
			# Check if node has valid moves and is not a dead end
			if Fnode[0] not in self.moves or Fnode[1] not in self.moves[Fnode[0]] or Fnode[2] not in self.moves[Fnode[0]][Fnode[1]]:
				continue
			# Expand to neighbours / Find nodes children
			for idx in self.moves[Fnode[0]][Fnode[1]][Fnode[2]]:
				delta = self.action[idx]
				neighbour = (Fnode[0]+delta[0], Fnode[1]+delta[1], Fnode[2]+delta[2])
				# Validate neighbour: if it is within bounds and not been expanded already
				if not self.inrange(neighbour) or str(neighbour) in visited:
					continue
				# Add to PQ and map node instance to parent instance
				# Mark the #instance in which the node has appeared as a neighbour
				if str(neighbour) not in counter:
					counter[str(neighbour)] = 1
				else:
					counter[str(neighbour)] +=1
				# Map to correct parent instance
				parent[str(neighbour) + 'x' + str(counter[str(neighbour)])] = F
				# Push to PriorityQueue
				if idx < 7:
					pq.heappush(Q, (cost + 10, str(neighbour) + 'x' + str(counter[str(neighbour)])))
				else:
					pq.heappush(Q, (cost + 14, str(neighbour) + 'x' + str(counter[str(neighbour)])))
			
		if not self.reached:
			return
		# Backtrace to get whole path from entry to exit
		node = F
		while node != str(root)+'x1':
			tmp = self.beautify(node)
			self.path.append(''.join(str(node).split('x')[0][1:-1].split(',')) + ' ' + self.get_s(tmp, self.beautify(parent[str(node)])))
			node = parent[str(node)]
		self.path.append(''.join(str(node).split('x')[0][1:-1].split(',')) + ' 0')
		self.n = len(self.path)
		

	def get_h(self, curr, goalstate):
		a = sorted([abs(curr[0]-goalstate[0]),abs(curr[1]-goalstate[1]),abs(curr[2]-goalstate[2])])
		dist = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])*10
		h = 0
		#14 * 3 * min{dx,dy,dz}/2
		v = a[0]
		h = int(h + int(v/2) * 42)
		if v%2==1:
			h = h + 24
		# 
		v = a[1] - a[0]
		h = int(h + int(v/2) * 28)
		if v%2==1:
			h += 14
		v = a[2] - a[1]
		h = int(h + 10*v)
		return h

	def astar(self):
		# initialize priority queue, visited set and parent map
		parent = {}
		counter = {}
		visited = set([])
		root = (self.x1, self.y1, self.z1)
		counter[str(root)] = 1
		Q = [(0,0,0,str(root) + 'x' + str(counter[str(root)]))]
		F, Fnode = None, None
		while not self.reached and len(Q) > 0:
			# Pop node with minimum cost
			front = Q.pop(0)
			old_f = front[0]
			cost = front[2]
			F = front[3]
			Fnode = self.beautify(F)
			if str(Fnode) in visited:
				continue
			# Check if goal
			if Fnode[0]==self.x2 and Fnode[1]==self.y2 and Fnode[2]==self.z2:
				self.score = cost
				self.reached = True
			
			visited.add(str(Fnode))
			if self.reached:
				continue
			# Check if node has valid moves
			if Fnode[0] not in self.moves or Fnode[1] not in self.moves[Fnode[0]] or Fnode[2] not in self.moves[Fnode[0]][Fnode[1]]:
				continue
			# Expand to neighbours
			for idx in self.moves[Fnode[0]][Fnode[1]][Fnode[2]]:
				delta = self.action[idx]
				neighbour = (Fnode[0]+delta[0], Fnode[1]+delta[1], Fnode[2]+delta[2])
				# validate neighbour
				if not self.inrange(neighbour) or str(neighbour) in visited:
					continue
				# add to pq and make parent entry
				if str(neighbour) not in counter:
					counter[str(neighbour)] = 1
				else:
					counter[str(neighbour)] +=1
				heuristic = self.get_h(neighbour, (self.x2, self.y2,self.z2))
				parent[str(neighbour) + 'x' + str(counter[str(neighbour)])] = F
				# use heuristic as a tiebreaker
				if idx < 7:
					pq.heappush(Q, (max(heuristic+cost+10, old_f),heuristic,cost+10, str(neighbour) + 'x' + str(counter[str(neighbour)])))
				else:
					pq.heappush(Q, (max(heuristic+cost+14, old_f), heuristic ,cost+14, str(neighbour) + 'x' + str(counter[str(neighbour)])))
			
		if not self.reached:
			return
		# Backtrace to get whole path from entry to exit
		node = F
		while node != str(root)+'x1':
			tmp = self.beautify(node)
			self.path.append(''.join(str(node).split('x')[0][1:-1].split(',')) + ' ' + self.get_s(tmp, self.beautify(parent[str(node)])))
			node = parent[str(node)]
			
		self.path.append(''.join(str(node).split('x')[0][1:-1].split(',')) + ' 0')		
		self.n = len(self.path)



	def launch(self):
		self.parse()
		#print(self.moves)
		if self.mode == "BFS":
			self.bfs()
		elif self.mode == "UCS":
			self.ucs()
		else:
			self.astar()

		if not self.reached:
			return [self.reached, "FAIL"]

		return [self.reached, self.score, self.n, self.path[::-1]]



def main():
	# Read input from file
	fn = '7'
	with open('input' + fn + '.txt', 'r') as f:
		inp = f.readlines()
	mode, dim, start, end = inp[0].strip(), inp[1].strip().split(), inp[2].strip().split(), inp[3].strip().split()
	# Initialize Bot
	bot = Search(mode, int(dim[0]), int(dim[1]), int(dim[2]), int(start[0]), int(start[1]), int(start[2]),
		int(end[0]), int(end[1]), int(end[2]),int(inp[4].strip()), inp[5:])
	# Initiate search
	result = bot.launch()
	# Parse output to file 
	ans = ""
	if result[0]:
		ans = ans + str(result[1]) + '\n' + str(result[2]) + '\n'
		for line in result[3]:
			ans = ans + line + '\n'
	else:
		ans = ans + 'FAIL\n'
	# print(ans[:-1])
	with open("output.txt", 'w') as g:
		g.write(ans)



if __name__=="__main__":
	main()
