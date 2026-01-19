// 1D mesh for subdomain 0: [0, gamma]
gamma = 0.75;
N = 20;
h = gamma / N;

Point(1) = {0, 0, 0, h};
Point(2) = {gamma, 0, 0, h};

Line(1) = {1, 2};

// Set tags to the boundaries.
// 0 = left (x=0), 1 = right (x=gamma, interface)
Physical Point(0) = {1};
Physical Point(1) = {2};

Physical Curve(10) = {1};

// Generate 1D mesh
Mesh 1;
Save "../mesh/mesh-problem-0.msh";