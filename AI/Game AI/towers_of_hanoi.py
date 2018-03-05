# Towers of Hanoi
# random solver (good to 9 rings)

# Extended: when an element goes onto the 3rd tower correctly freeze
#		    it there, so it cannot be shuffled along again.

import random

def move_ring(towers, from_tower, to_tower):
	towers[to_tower].append(towers[from_tower][-1])
	towers[from_tower].pop()

def towers_with_rings(l_towers, r):
	l_towers_with_rings = []
	l_check = [i for i in range(r, 0, -1)]
	for i, towers in enumerate(l_towers):
		if len(towers) > 0:	
			l_towers_with_rings.append(i)
	# if the third tower is an exact subset of the end solution third
	# tower, then do not select it as the from_tower...
	if l_towers[2] == l_check[:len(l_towers[2])] and 2 in l_towers_with_rings:
		l_towers_with_rings.remove(2)
	return l_towers_with_rings

def solve(r):
	
	# initial state
	l_towers = [
		[i for i in range(r, 0, -1)],
		[],
		[]]

	num_iters = 0
	while len(l_towers[2]) < r:
		num_iters += 1
		random_from_tower = random.choice(towers_with_rings(l_towers, r))
		random_to_tower = random.randint(0, 2)
		if random_from_tower == random_to_tower:
			continue

		# check valid move
		if len(l_towers[random_to_tower]) == 0:
			move_ring(l_towers, random_from_tower, random_to_tower)
		elif l_towers[random_from_tower][-1] < l_towers[random_to_tower][-1]:
			move_ring(l_towers, random_from_tower, random_to_tower)
		else:
			continue

		for tower in l_towers:
			print tower
		print '\n'

	return num_iters
print solve(5)


##################################################
# Procedural proper solution
##################################################

'''
Using
http://www.algomation.com/algorithm/towers-hanoi-recursive-visualization
I think I've found a pattern for r=5:
- you want n - (n-1) rings on tower 3
- then n - (n-2) rings on tower 2
- then n - (n-3) rings on tower 3
- then n - (n-4) rings on tower 2 (or n-1 rings)
(i.e.
1 ring on tower 3
2 rings on tower 2
3 rings on tower 3
4 rings on tower 2
)


EB noticed that there is a pattern in the moves!
It goes:

For r = 5
1 2 1 
3
1 2 1
4
1 2 1
3
1 2 1
5
# the second pole is full so..
# then just reply the moves that have been done in reverse order!
1 2 1
3 
1 2 1 
4
1 2 1
3
1 2 1

For r = 4
1 2 1
3
1 2 1
4
1 2 1
3
1 2 1
# the second pole is full so..
# then just reply the moves that have been done in reverse order!

Can a combination of this and the above form an algorithm?



Another hypothesis that could be the final part of the puzzle in combination
with the above two:

On the tower 3 ones, i.e. "3 rings on tower 3", the 1 2 1 sequence always starts
with the first 1 ring going on the largest ring possible.. (?)

On the tower 2 ones, i.e. "4 rings on tower 2", the 1 2 1 sequence always starts
with the first 1 ring going on the smallest ring possible.. (?)

And if there is an empty tower for the first 1 in the 1 2 1 sequence it should
always move to the empty tower..

Treat each peg as a seperate thingy- 
Peg 1 pattern = 
54321
5432
543
54
541
54
5
52
521
52
5

Peg 2 pattern=


------
When there is a 1 2 1 slot and I'm trying to make a tower I could store an array
of moves to see which does it quicker?



'''