#!/bin/bash

# Script per generare mesh con diverse dimensioni h

for h in 0.1 0.05 0.025 0.0125; do
    cat > mesh-square-h${h}.geo << EOF
a = 0.0; // Lower bound.
b = 1.0; // Upper bound.

h = ${h}; // Mesh size.

L = b - a; // Length of the square side.

// Create one point in the origin.
Point(1) = {a, a, 0, h};

// Extrude the point along x to create one side.
Extrude {L, 0, 0} { Point{1}; Layers{L / h}; }

// Extrude that side along y to create the square.
Extrude {0, L, 0} { Line{1}; Layers{L / h}; }

// Define the tags.
Physical Line(0) = {3};  // x = 0
Physical Line(1) = {4};  // x = 1
Physical Line(2) = {1};  // y = 0
Physical Line(3) = {2};  // y = 1

Physical Surface(10) = {5};

// Generate a 2D mesh.
Mesh 2;

// Save mesh to file.
Save "mesh-square-h${h}.msh";
EOF

    gmsh -2 mesh-square-h${h}.geo -format msh4
    rm mesh-square-h${h}.geo
    echo "Generated mesh-square-h${h}.msh"
done
