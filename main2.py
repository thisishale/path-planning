# In[]: 
#this function finds (pi-theta): in which theta is the angle line.
def azimuth(point1, point2):
    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return np.degrees(angle)
#getangle function finds the angle of intersection between two lines.
def getangle(p1,inter,p2):
    out=azimuth(inter,p2)-azimuth(inter,p1)
    #all angles are positive:
    if out<0:
        out=out+360
    return out
#change refrenece:
def change_ref(mat,phi,s):
    new_mat=np.matmul([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]],mat)+(s)
    return(new_mat)
#change reference back:
def change_ref_back(new_mat,phi,s):
    changed_back=np.matmul(np.linalg.inv([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]),np.subtract(new_mat,s))
    return np.matrix.round(changed_back) 

#this function finds the distance between a line and obstacles and returns the 
#distance between the line and the nearest obstacle.
def getmindistance(line,polygons):
    distances=[]
    alpha=-1
    for shape in polygons:
        distances.append(np.exp(alpha*line.distance(shape)))
    return min(distances)
#get the linesequences.
def get_linesequence(gen,pop):
    indices=[]
    linesequence=[]
    linesequence_row=[]
    for i in range(pop):
        indices_row=[]
        for j in range(num_of_points-1):
            indices_row.append(np.random.randint(len(gen[j])))
        indices.append(indices_row)
    for i in range(pop):
        linesequence_row=[]
        for j in range(num_of_points-1):
            if j==0:
                linesequence_row.append(np.array(s))
            linesequence_row.append(gen[j][indices[i][j]])
            if j==num_of_points-2:
                linesequence_row.append(np.array(t))
        linesequence.append(linesequence_row)
    return linesequence
#gets the lines out of linesequences made before.also in this function the 
#number of collisions for each line and the lines with no collisions are returned. 
def getlines_cols(linesequence,polylists):
    lines=[]
    no_col_lines=[] 
    pop=len(linesequence)
    count=np.zeros((pop,1))
    col=np.zeros((pop,1))
    nointersect=np.zeros((pop,1))   
    for i in range(pop):
        lines.append(shapely.geometry.LineString(linesequence[i]))
    for i in range(pop):
        for j in range(len(polylists)):
            if lines[i].intersects(polylists[j])==False:
                count[i]=count[i]+1
            else:
                col[i]=col[i]+1
        if count[i]==len(polylists):
            no_col_lines.append(lines[i])
            nointersect[i]=1
        else:
            nointersect[i]=0
    return lines,col,no_col_lines
#this function finds the intersection with most rotation and returns pi-intersection angle
def get_max_angle(linesequence):
    partial_angle=[]
    for i in range(1,num_of_points):
        partial_angle.append(180-getangle(linesequence[i-1],linesequence[i],linesequence[i+1]))
    return max(partial_angle)
#this function calculates the criterions including distance from nearest obstacle,
#length of path and smoothness of path.
def get_citerions(pop, lines,linesequence):
    length=np.zeros((len(lines)))
    for i in range(len(lines)):
        length[i]=(lines[i].length)
    distance=np.zeros((len(lines)))
    for i in range(len(lines)):
        distance[i]=(getmindistance(lines[i],polylists))
    angle=np.zeros((len(lines)))
    for i in range(len(lines)):
        partial_angle=[]
        for j in range(1,num_of_points-1):
            partial_angle.append(180-getangle(linesequence[i][j-1],linesequence[i][j],linesequence[i][j+1]))
        angle[i]=(max(partial_angle))
    return length,distance,angle
#this function computes cost based on  criterions plus the number of collisions 
#in each line. length and angle are also normalized. each criterion has a weight 
#which indicated the importance of it. since the collision is the most important 
#critertion here, it has the biggest weight.
def get_cost(length,distance,angle,col):
    normalized_distance=np.reshape(distance,(-1,1))
    normalized_length=np.reshape(length/full_distance,(-1,1))
    normalized_angle=np.reshape(angle/180,(-1,1))
    if max(col)==0:
        normalized_col=0
    else:
        normalized_col=np.reshape(col/max(col),(-1,1))
    cost= 2*normalized_distance+ normalized_length+ 2*normalized_angle+ 10*normalized_col
    return cost
#this function gets a line which is the crossover result from two random lines.
#for crossover two random lines are selected, then for each point a probability 
#is generated, if the probability is less than a threshhold, then the crossover
#happens at that point of line. the x and y for each point in two lines gets combined
#with weights.
def get_crossed(linesequence):
    r=np.random.randint(len(linesequence), size=2)
    p=np.random.rand(1,num_of_points+1) 
    th=0.6
    new_linesequence_row=[]
    for i in range(num_of_points+1):
        if p[0][i]<th:
            new_linesequence_row.append(np.array(linesequence[r[0]][i])*0.4+np.array(linesequence[r[1]][i])*0.6)
        else:
            new_linesequence_row.append(linesequence[r[0]][i])
    return new_linesequence_row 
#in this segment, mutation is implemented. mutation only uses on parent which is chosen 
#randomly. choosing which point in a line is simillar to the crossover section.
def mutate(linesequence,d):
    r=np.random.randint(len(linesequence), size=1)
    plus=np.random.randint(2, size=1)
    p=np.random.rand(1,num_of_points+1) 
    th=0.5
    var=np.array([0,50])
    linesequence_row=[]
    for i in range(num_of_points+1):
        if p[0][i]<th and plus[0]==1:
            linesequence_row.append(change_ref(change_ref_back(np.array(linesequence[r[0]][i]),phi,s)+(var/d),phi,s))
        elif p[0][i]<th and plus==0:
            linesequence_row.append(change_ref(change_ref_back(np.array(linesequence[r[0]][i]),phi,s)-(var/d),phi,s))
        else:
            linesequence_row.append(np.array(linesequence[r[0]][i]))
    return linesequence_row
# In[]:
import numpy as np
import matplotlib.pyplot as plt
plt.clf()
plt.close("all")
# In[]: 
#in this part polygons(obstacles) are defined using shapely library.
import descartes
import shapely.geometry
import numpy as np
poly1 = shapely.geometry.Polygon([[10, 20], [15, 20], [16, 26],[12,25]])
poly2 = shapely.geometry.Polygon([[25, 30], [32, 30], [33, 40],[26,38]])
poly3 = shapely.geometry.Polygon([[35, 20], [40, 22], [42, 26],[35,25]])
poly4 = shapely.geometry.Polygon([[20, 5], [30, 5], [30, 16],[20,15]])
poly5 = shapely.geometry.Polygon([[5, 35], [15, 35], [15, 45],[5,47]])
poly6 = shapely.geometry.Polygon([[5, 5], [15, 7], [15, 15],[5,17]])
#listng shapes.
polylists=[poly1, poly2, poly3, poly4, poly5, poly6]

# In[2]:
import math as m
#start:
s=[1,4]
#target:
t=[45,30]
#end of cartesian enviroment.
x_end=50
y_end=50
#coordinates rotation
phi=np.arctan((t[1]-s[1])/(t[0]-s[0]))
#in this part the first generation of points that can be chosen as linesequencs
#is generated randomly.
first_gen_y=[]
first_gen_x=[]
first_gen=[]
#random points on each of those parallel lines.(this part can be done with 
#random function too.) 
num_of_rands=100
#in this part the number of segments in which the points are chosen is defined.
num_of_points=8
#first population is made of 'first_pop' lines.
first_pop=50
new_firstgen=[]
np.random.seed(1)
#distance between start and target.
full_distance=m.sqrt(((t[0]-s[0])**2)+((t[1]-s[1])**2))
mini_distance=full_distance/num_of_points
for i in range(1,num_of_points):
    first_gen_row=[]
    first_gen_y.append((np.random.rand(1,num_of_rands)-0.5)*100)
    first_gen_x.append(i*mini_distance)
    for j in range(len(first_gen_y[i-1][0])):
        temp=change_ref([first_gen_x[i-1],first_gen_y[i-1][0][j]],phi,s)
        a=shapely.geometry.Point(temp)
        in_obs=[]
        for k in polylists:
            in_obs.append(a.within(k))
            #in this part the generation of points are created but with a few conditions:
            #*the points must be in the first coordinate plane.
            #*the points should not be inside obstacles.
        if temp[0]>0 and temp[1]>0 and temp[0]<x_end and temp[1]<y_end and sum(in_obs)==0:
            first_gen_row.append(temp)
    first_gen.append(first_gen_row)
# In[]
cost=[]
linesequence=get_linesequence(first_gen,first_pop)
[lines,col,no_col_lines]=getlines_cols(linesequence,polylists)
[length,distance,angle]=get_citerions(first_pop, lines,linesequence)
cost.extend(get_cost(length,distance,angle,col))
sorted_cost=sorted(cost)
# In[]
d=1
costlist=[]
it=100
save=[]
num_of_cols=[]
count=0
stop=False
#first crossover is run, then mutation for only one path happens.both in one iteration.
while stop==False:
    for j in range(50):
        linesequence.append(get_crossed(linesequence))
    [lines,col,no_col_lines]=getlines_cols(linesequence,polylists)
    [length,distance,angle]=get_citerions(len(linesequence), lines,linesequence)   
    cost=get_cost(length,distance,angle,col)    
    sorted_cost=sorted(cost)
    sorted_index=sorted(range(len(cost)), key=lambda k: cost[k])  
    costlist.append(sorted_cost[0])
    num_of_cols.append(col[sorted_index[0]])
    if costlist[-1]<1:
        save.append(linesequence[sorted_index[0]])
    x=[]
    for k in range(first_pop):
        x.append(linesequence[sorted_index[k]])
    linesequence=[]
    cost=[]
    linesequence.extend(x)
    d=d+5
    linesequence.append(mutate(linesequence,d))
    [lines,col,no_col_lines]=getlines_cols(linesequence,polylists)
    [length,distance,angle]=get_citerions(len(linesequence), lines,linesequence)   
    cost=get_cost(length,distance,angle,col)    
    sorted_index=sorted(range(len(cost)), key=lambda k: cost[k])   
    sorted_cost=sorted(cost)
    x=[]
    for t in range(first_pop):
        x.append(linesequence[sorted_index[t]])
    linesequence=[]
    cost=[]
    linesequence.extend(x) 
    if count>50:
        if costlist[-20]-costlist[-1]<0.001:
            stop=True
    #after every 20 iteration, all paths are plotted.
    if count%20==0:
        plt.xlim([0,50])
        plt.ylim([0,50])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(*np.array(linesequence).T, color='blue', linewidth=1, solid_capstyle='round')
        ax.add_patch(descartes.PolygonPatch(poly1, fc='blue', alpha=0.5))
        ax.add_patch(descartes.PolygonPatch(poly2, fc='blue', alpha=0.5))
        ax.add_patch(descartes.PolygonPatch(poly3, fc='blue', alpha=0.5))
        ax.add_patch(descartes.PolygonPatch(poly4, fc='blue', alpha=0.5))
        ax.add_patch(descartes.PolygonPatch(poly5, fc='blue', alpha=0.5))
        ax.add_patch(descartes.PolygonPatch(poly6, fc='blue', alpha=0.5))
        ax.axis('equal')
        plt.grid(color='b', linestyle='-', linewidth=0.1)
        plt.title('all routes in iteration %i'%count)
        plt.xlabel('x')
        plt.ylabel('y')

    count=count+1
    

# In[]
#plot the minimum cost in each iteration.
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.xlim([0,count]) 
plt.plot(range(count),np.array(costlist))   
plt.xlabel('iteration')
plt.ylabel('cost for best route')
#plot the number of collisions for best path in each iteration.
plt.subplot(2, 1, 2)
plt.xlim([0,count])   
plt.plot(range(count),num_of_cols)
plt.xlabel('iteration')
plt.ylabel('collisions for best route')
# In[]
#plot the best path.
plt.xlim([0,50])
plt.ylim([0,50])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(*np.array(linesequence[0]).T, color='blue', linewidth=1, solid_capstyle='round')
ax.add_patch(descartes.PolygonPatch(poly1, fc='blue', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(poly2, fc='blue', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(poly3, fc='blue', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(poly4, fc='blue', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(poly5, fc='blue', alpha=0.5))
ax.add_patch(descartes.PolygonPatch(poly6, fc='blue', alpha=0.5))
ax.axis('equal')
plt.grid(color='b', linestyle='-', linewidth=0.1)