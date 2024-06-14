#version 150

in vec2 tex;
out vec4 c;

uniform sampler2D textureImage;

void main()
{
  // compute the final pixel color
  c = texture(textureImage, tex);
}