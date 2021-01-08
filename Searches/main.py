import heapq as pq

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


	def ucs(self):
		# initialize priority queue, visited set and parent map
		parent = {}
		visited = set([])
		Q = [(0, (self.x1, self.y1, self.z1))]
		while not self.reached and len(Q) > 0:
			# Pop node with minimum cost
			front = Q.pop(0)
			cost = front[0]
			F = front[1]
			if str(F) in visited:
				continue;
			# Check if goal
			if F[0]==self.x2 and F[1]==self.y2 and F[2]==self.z2:
				self.score = cost
				self.reached = True
			
			visited.add(str(F))
			if self.reached:
				continue
			# Check if node has valid moves
			if F[0] not in self.moves or F[1] not in self.moves[F[0]] or F[2] not in self.moves[F[0]][F[1]]:
				continue
			# Expand to neighbours
			for idx in self.moves[F[0]][F[1]][F[2]]:
				delta = self.action[idx]
				neighbour = (F[0]+delta[0], F[1]+delta[1], F[2]+delta[2])
				# validate neighbour
				if not self.inrange(neighbour) or str(neighbour) in visited:
					continue
				# add to pq and make parent entry
				if idx < 7:
					parent[str(neighbour)] = F
					pq.heappush(Q, (cost + 10, neighbour))
				else:
					parent[str(neighbour)] = F
					pq.heappush(Q, (cost + 14, neighbour))
			
		if not self.reached:
			return
		# Backtrace to get whole path from entry to exit
		node = (self.x2, self.y2, self.z2)
		while node != (self.x1, self.y1, self.z1):
			self.path.append(''.join(str(node)[1:-1].split(',')) + ' ' + self.get_s(node, parent[str(node)]))
			node = parent[str(node)]

		self.path.append(''.join(str(node)[1:-1].split(',')) + ' 0')
		self.n = len(self.path)

	def get_h(self, curr, next):
		a = sorted([abs(curr[0]-next[0]),
			abs(curr[1]-next[1]),
			abs(curr[2]-next[2])])
		return 10*a[2] + 4*a[1] - 3*a[0]

	def astar(self):
		# initialize priority queue, visited set and parent map
		parent = {}
		visited = set([])
		Q = [(0,0, (self.x1, self.y1, self.z1))]
		while not self.reached and len(Q) > 0:
			# Pop node with minimum cost
			front = Q.pop(0)
			# print(front)
			old_f = front[0]
			cost = front[1]
			F = front[2]
			if str(F) in visited:
				continue;
			# Check if goal
			if F[0]==self.x2 and F[1]==self.y2 and F[2]==self.z2:
				self.score = cost
				self.reached = True
			
			visited.add(str(F))
			if self.reached:
				continue
			# Check if node has valid moves
			if F[0] not in self.moves or F[1] not in self.moves[F[0]] or F[2] not in self.moves[F[0]][F[1]]:
				continue
			# Expand to neighbours
			for idx in self.moves[F[0]][F[1]][F[2]]:
				delta = self.action[idx]
				neighbour = (F[0]+delta[0], F[1]+delta[1], F[2]+delta[2])
				# validate neighbour
				if not self.inrange(neighbour) or str(neighbour) in visited:
					continue
				# add to pq and make parent entry
				heuristic = self.get_h(neighbour, (self.x2, self.y2,self.z2))
				if idx < 7:
					parent[str(neighbour)] = F
					pq.heappush(Q, (max(old_f, heuristic+cost+10), cost+10, neighbour))
				else:
					parent[str(neighbour)] = F
					pq.heappush(Q, (max(old_f, heuristic+cost+14), cost+14, neighbour))
			
		if not self.reached:
			return
		# Backtrace to get whole path from entry to exit
		pre = None
		node = (self.x2, self.y2, self.z2)
		while node != (self.x1, self.y1, self.z1):
			self.path.append(''.join(str(node)[1:-1].split(',')) +' ' + self.get_s(node, parent[str(node)]))
			node = parent[str(node)]
			
		self.path.append(''.join(str(node)[1:-1].split(',')) + ' 0')
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
	with open("input.txt", 'r') as f:
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
