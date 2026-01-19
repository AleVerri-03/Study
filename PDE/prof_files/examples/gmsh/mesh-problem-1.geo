// 1D mesh for subdomain 1: [gamma, 1]
gamma = 0.75;
N = 20;
h = (1.0 - gamma) / N;

Point(1) = {gamma, 0, 0, h};
Point(2) = {1, 0, 0, h};

Line(1) = {1, 2};

// Set tags to the boundaries.
// 0 = left (x=gamma, interface), 1 = right (x=1)
Physical Point(0) = {1};
Physical Point(1) = {2};

Physical Curve(10) = {1};

// Generate 1D mesh
Mesh 1;
Save "../mesh/mesh-problem-1.msh";