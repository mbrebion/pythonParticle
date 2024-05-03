circlesVertex = """
#version 330
attribute vec2 position;
attribute float color;
uniform float radius;
uniform vec2 resolution;
uniform vec2 spaceLength;

varying vec2 v_center;
varying float v_radius;
varying vec3 v_color;

void main () {
    v_center = position / spaceLength * resolution;
    v_radius = radius;
    v_color = vec3(0.,0.,0.);
    if (color <0.0){
        v_color = vec3(0.,0.,sqrt(-color));
    }else{
        v_color = vec3(sqrt(color),0.,0.);
    }


    gl_Position = vec4(2*position/spaceLength-1.0, 0.0, 1.0); 
    gl_PointSize = 3.0 + ceil(2.0*radius);    
    }
"""

circlesFragment = """
#version 330
varying vec2 v_center;
varying float v_radius;
varying vec3 v_color;

void main() {
    vec2 p = gl_FragCoord.xy - v_center;
    float a = 1.0;
    float d = length(p)-v_radius*0.99;
    if(d > 0.0) a = exp(-4*d*d);
    gl_FragColor = vec4(v_color, a);    
}
"""
