## Instructions
Suppose you are controlling a robot to perform manipulation tasks by writing constraint functions in Python. The manipulation task is given as an image of the environment, overlayed with keypoints marked with their indices, along with a text instruction. For each given task, please perform the following steps:
- Determine how many stages are involved in the task. Grasping must be an independent stage. Some examples:
  - "pouring tea from teapot":
    - 3 stages: "grasp teapot", "align teapot with cup opening", and "pour liquid"
  - "put red block on top of blue block":
    - 3 stages: "grasp red block", "drop the red block on top of blue block"
  - "reorient bouquet and drop it upright into vase":
    - 3 stages: "grasp bouquet", "reorient bouquet", and "keep upright and drop into vase"
- For each stage, write two kinds of constraints, "sub-goal constraints" and "path constraints". The "sub-goal constraints" are constraints that must be satisfied **at the end of the stage**, while the "path constraints" are constraints that must be satisfied **within the stage**. Some examples:
  - "pouring liquid from teapot":
    - "grasp teapot" stage:
      - 1 sub-goal constraints: "align the end-effector with the teapot handle"
      - 0 path constraints
    - "align teapot with cup opening" stage:
      - 1 sub-goal constraints: "the teapot spout needs to be 10cm above the cup opening"
      - 2 path constraints: "the robot must still be grasping the teapot handle", "the teapot must stay upright to avoid spilling"
    - "pour liquid" stage:
      - 2 sub-goal constraints: "the teapot spout needs to be 5cm above the cup opening", "the teapot spout must be tilted to pour liquid"
      - 2 path constraints: "the robot must still be grasping the teapot handle", "the teapot spout is directly above the cup opening"
  - "put red block on top of blue block":
    - "grasp red block" stage:
      - 1 sub-goal constraints: "align the end-effector with the red block"
      - 0 path constraints
    - "drop the red block on top of blue block" stage:
      - 1 sub-goal constraints: "the red block is 10cm on top of the blue block"
      - 1 path constraints: "the robot must still be grasping the red block"
  - "eorient bouquet and drop it upright into vase":
    - "grasp bouquet" stage:
      - 1 sub-goal constraints: "align the end-effector with the bouquet stem"
      - 0 path constraints
    - "reorient bouquet" stage:
      - 1 sub-goal constraints: "the bouquet is upright (parallel to the z-axis)"
      - 1 path constraints: "the robot must still be grasping the bouquet stem"
    - "keep upright and drop into vase" stage:
      - 2 sub-goal constraints: "the bouquet must still stay upright (parallel to the z-axis)", "the bouquet is 20cm above the vase opening"
      - 1 path constraints: "the robot must still be grasping the bouquet stem"
- Summarize keypoints to be grasped in all grasping stages by defining the `grasp_keypoints` variable.
- Summarize at the end of which stage the robot should release the keypoints by defining the `release_keypoints` variable.

**Note:**
- Each constraint takes a dummy end-effector point and a set of keypoints as input and returns a numerical cost, where the constraint is satisfied if the cost is smaller than or equal to zero.
- For each stage, you may write 0 or more sub-goal constraints and 0 or more path constraints.
- Avoid using "if" statements in your constraints.
- Avoid using path constraints when manipulating deformable objects (e.g., clothing, towels).
- You do not need to consider collision avoidance. Focus on what is necessary to complete the task.
- Inputs to the constraints are as follows:
  - `end_effector`: np.array of shape `(3,)` representing the end-effector position.
  - `keypoints`: np.array of shape `(K, 3)` representing the keypoint positions.
- For any path constraint that requires the robot to be still grasping a keypoint `i`, you may use the provided function `get_grasping_cost_by_keypoint_idx` by calling `return get_grasping_cost_by_keypoint_idx(i)` where `i` is the index of the keypoint. 
- Inside of each function, you may use native Python functions, any NumPy functions, and the provided `get_grasping_cost_by_keypoint_idx` function.
- For grasping stage, you should only write one sub-goal constraint that associates the end-effector with a keypoint. No path constraints are needed.
- In order to move a keypoint, its associated object must be grasped in one of the previous stages.
- The robot can only grasp one object at a time.
- Grasping must be an independent stage from other stages.
- You may use two keypoints to form a vector, which can be used to specify a rotation (by specifying the angle between the vector and a fixed axis).
- You may use multiple keypoints to specify a surface or volume.
- The keypoints marked on the image start with index 0, same as the given argument `keypoints` array.
- For a point `i` to be relative to another point `j`, the function should define an `offsetted_point` variable that has the delta added to keypoint `j and then calculate the norm of the xyz coordinates of the keypoint `i` and the `offsetted_point`.
- If you would like to specify a location not marked by a keypoint, try using multiple keypoints to specify the location (e.g., you may take the mean of multiple keypoints if the desired location is in the center of those keypoints).

**Structure your output in a single python code block as follows:**
```python

# Your explanation of how many stages are involved in the task and what each stage is about.
# ...

num_stages = ?

### stage 1 sub-goal constraints (if any)
def stage1_subgoal_constraint1(end_effector, keypoints):
    """Put your explanation here."""
    ...
    return cost
# Add more sub-goal constraints if needed
...

### stage 1 path constraints (if any)
def stage1_path_constraint1(end_effector, keypoints):
    """Put your explanation here."""
    ...
    return cost
# Add more path constraints if needed
...

# repeat for more stages
...

"""
Summarize keypoints to be grasped in all grasping stages.
The length of the list should be equal to the number of stages.
For grapsing stage, write the keypoint index. For non-grasping stage, write -1.
"""
grasp_keypoints = [?, ..., ?]

"""
Summarize at **the end of which stage** the robot should release the keypoints.
The keypoint indices must appear in an earlier stage as defined in `grasp_keypoints` (i.e., a keypoint can only be released only if it has been grasped previously).
Only release object when it's necessary to complete the task, e.g., drop bouquet in the vase.
The length of the list should be equal to the number of stages.
If a keypoint is to be released at the end of a stage, write the keypoint index at the corresponding location. Otherwise, write -1.
"""
release_keypoints = [?, ..., ?]

```

## Query
Query Task: "{instruction}"
Query Image: