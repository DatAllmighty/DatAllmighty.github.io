import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls'
import Stats from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/stats.module'
import { GPUComputationRenderer } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/misc/GPUComputationRenderer.js';

var parent = document.getElementById("embedded-video-div")

//Create rendering context
const renderer = new THREE.WebGLRenderer()
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(parent.clientWidth, parent.clientHeight)
parent.appendChild(renderer.domElement)
const stats = Stats()
parent.appendChild(stats.dom)

//Create camera & controls, forced render camera for screenspace quad
const terrainHeightScale = 25;
const camera = new THREE.PerspectiveCamera(
    75,
    parent.clientWidth / parent.clientHeight,
    0.01,
    1000
)
camera.position.y =  16.55
camera.position.z = 3
camera.position.x = 6.6

const renderCamera = new THREE.OrthographicCamera( -1, 1, 1, -1, 0, 1 )
var controls = new OrbitControls( camera, parent )
controls.enabled = false;

var clock = new THREE.Clock();


//Set resize callbacks
window.addEventListener('resize', onWindowResize, false)
function onWindowResize() {
    camera.aspect = parent.clientWidth / parent.clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(parent.clientWidth, parent.clientHeight)
}


//Create scene
const scene = new THREE.Scene()
scene.add(new THREE.AxesHelper(5))

//Shaders
const vertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
    }
`
const fragmentShader = `
#define FLT_MAX 3.402823466e+38F   
#define ACCURACY_THRESH 0.01

varying vec2 vUv;

//0 = Directional, 1=point
struct Light
{	
	vec3 position;
	vec3 direction;
	vec3 intensity;
	vec3 constant_attenuation;
	vec3 linear_attenuation;
	vec3 quadratic_attenuation;
	vec3 params;
};

//All materials have phong params
struct Material
{	
	vec3 diffuse_color;
	vec3 specular_color;
	vec3 reflective_color;
	vec3 transparent_color;
	vec3 params;
};

//params (radius, null null)
//params_i (material, transform, null)
struct Sphere
{
	vec3 center;
	vec3 params;
	vec3 params_i;
};

//params_i (material, transform, null)
struct Box
{
	vec3 vmin;
	vec3 vmax;
	vec3 params_i;
};


struct SkyboxParams {
    vec3 vmin_s;
	vec3 vmax_s;
	vec3 params_i_s;
};

uniform Sphere spheres[1];

uniform Material materials[1];

uniform Light lights[1];

uniform SkyboxParams skyboxIn;

uniform vec3 scene_ambient_light;
uniform vec3 scene_background_color;
uniform vec3 camera_center;
uniform vec3 camera_direction;
uniform vec3 camera_up;
uniform vec3 camera_horizontal;

uniform vec2 iResolution;

uniform float camera_fov;
uniform float camera_tmin;

uniform int light_count;
uniform int sphere_count;
uniform int	has_skybox;

uniform int	samples;

uniform sampler2D skyboxTex;
uniform sampler2D heightTex;
uniform sampler2D normalTex;
uniform sampler2D diffuseTex;
uniform sampler2D htTex;
uniform float heightScale;
uniform float gridSize;

Box skybox;

struct Ray {
	vec3 origin;
	vec3 direction;
};

struct Hit {
	float t;
	vec3 normal;
	int material;
};

vec3 PointAtParameter(Ray r, float t) {
	return r.origin + r.direction * t;
}

void GetIncidentIllumination(Light light, Ray r, Hit hit, out vec3 incident_intensity, out vec3 dir_to_light, out float distance) {
	//Directional light
	if (light.params.x < 0.1) {
		dir_to_light = -normalize(light.direction);
		incident_intensity = light.intensity;
		distance = FLT_MAX;
	}
	else if (light.params.x > 0.1) {
		vec3 pp = light.position - PointAtParameter(r, hit.t);
		distance = length(pp);
		incident_intensity = (1.f / (light.quadratic_attenuation * pow(distance, 2.0) + light.linear_attenuation * pow(distance, 1.0) + light.constant_attenuation)) * light.intensity;
		dir_to_light = pp/distance;
	}
}

bool SphereIntersection(Sphere sp, Ray r, float tmin, inout Hit hit) {

	vec3 tmp = sp.center - r.origin;
	vec3 dir = r.direction;
	float radius = sp.params.x;
	float A = dot(dir, dir);
	float B = -2.0 * dot(dir, tmp);
	float C = dot(tmp, tmp) -  radius * radius;
	float radical = B * B - 4.0 * A * C;
	if (radical < 0.0f) {
		return false;
	}

	radical = sqrt(radical);
	float t_m = (-B - radical) / (2.0 * A);
	float t_p = (-B + radical) / (2.0 * A);
	vec3 pt_m = PointAtParameter(r, t_m);
	vec3 pt_p = PointAtParameter(r, t_p);

	bool flag = t_m <= t_p;
	if (!flag) {
		return false;
	}
	// choose the closest hit in front of tmin
	float t = (t_m < tmin) ? t_p : t_m;

	if (hit.t > t && t > tmin) {
		vec3 normal = normalize(PointAtParameter(r, t) - sp.center);
		hit.t = t;
		hit.material = int(sp.params_i.x);
		hit.normal = normal;
		return true;
	}
	return false;
}


bool BoxIntersection(Box b, Ray r, float tmin, inout Hit hit) {
	
	if (r.direction.x == 0.0f && (r.origin.x < min(b.vmin.x, b.vmax.x) || r.origin.x > max(b.vmin.x, b.vmax.x))) {
		return false;
	}

	float tstart = r.direction.x == 0.0f ? tmin : (b.vmin.x - r.origin.x) / r.direction.x;
	float tend = r.direction.x == 0.0f ? FLT_MAX : (b.vmax.x - r.origin.x) / r.direction.x;

	if (tstart > tend) {
		float temp = tend;
		tend = tstart;
		tstart = temp;
	}

	if (r.direction.y == 0.0f && (r.origin.y < min(b.vmin.y, b.vmax.y) || r.origin.y > max(b.vmin.y, b.vmax.y))) {
		return false;
	}

	float tymin = r.direction.y == 0.0f ? tmin : (b.vmin.y - r.origin.y) / r.direction.y;
	float tymax = r.direction.y == 0.0f ? FLT_MAX : (b.vmax.y - r.origin.y) / r.direction.y;


	if (tymin > tymax) {
		float temp = tymin;
		tymin = tymax;
		tymax = temp;
	}

	tstart = max(tymin, tstart);
	tend = min(tymax, tend);

	if (tstart > tend || tend < tmin) {
		return false;
	}

	if (r.direction.z == 0.0f && (r.origin.z < min(b.vmin.z, b.vmax.z) || r.origin.z > max(b.vmin.z, b.vmax.z))) {
		return false;
	}

	float tzmin = r.direction.z == 0.0f ? tmin : (b.vmin.z - r.origin.z) / r.direction.z;
	float tzmax = r.direction.z == 0.0f ? FLT_MAX : (b.vmax.z - r.origin.z) / r.direction.z;

	if (tzmin > tzmax) {
		float temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
	}

	tstart = max(tzmin, tstart);
	tend = min(tzmax, tend);

	if (tstart > tend || tend < tmin) {
		return false;
	}
	if (tstart > tmin && tstart < hit.t) {
		hit.t = tstart;
	}
	else if (tend < hit.t) {
		hit.t = tend;

	}
	else {
		return false;
	}
	hit.material = int(b.params_i.x);

	vec3 p = PointAtParameter(r, hit.t);

	if (abs(p.z - b.vmin.z) < 0.0001f) {
		hit.normal = vec3(0.0f, 0.0f, -1.0f);
	}

	else if (abs(p.z - b.vmax.z) < 0.0001f) {
		hit.normal = vec3(0.0f, 0.0f, 1.0f);
	}

	else if (abs(p.y - b.vmin.y) < 0.0001f) {
		hit.normal = vec3(0.0f, -1.0f, 0.0f);
	}

	else if (abs(p.y - b.vmax.y) < 0.0001f) {
		hit.normal = vec3(0.0f, 1.0f, 0.0f);
	}

	else if (abs(p.x - b.vmin.x) < 0.0001f) {
		hit.normal = vec3(-1.0f, 0.0f, 0.0f);
	}

	else if (abs(p.x - b.vmax.x) < 0.0001f) {
		hit.normal = vec3(1.0f, 0.0f, 0.0f);
	}
	
	return true;
}


bool TraceShadow(Ray r, Hit h, float tmin) {
	//Sphere
	bool intersection = false;
	
	for (int i = 0; i < sphere_count; i++) {
		bool b = SphereIntersection(spheres[i], r, tmin, h);
		if (b && h.t > tmin) {
			return true;
		}
	}

	return false;
}

vec3 Shade(Hit hit, Ray r, Material mat) {
	
	vec3 answer = mat.diffuse_color * scene_ambient_light;

	for (int i = 0; i < light_count; i++) {
		Light l = lights[i];
		vec3 incident_intensity;
		vec3 dir_to_light;
		float distance;
		GetIncidentIllumination(l, r, hit, incident_intensity, dir_to_light, distance);
		vec3 di = incident_intensity * clamp(dot(hit.normal, dir_to_light), 0.0f, 1.0f) * mat.diffuse_color;
		vec3 ri = normalize(dir_to_light - 2.f * dot(dir_to_light, hit.normal) * hit.normal);
		vec3 si = vec3(0.f, 0.f, 0.f);
		if (dot(dir_to_light, hit.normal) > 0.0f) {
			si = incident_intensity * mat.specular_color * pow(clamp(dot(r.direction, ri), 0.0f, 1.0f), mat.params.y);
		}
		vec3 origin = PointAtParameter(r, hit.t) + 0.0001f * hit.normal;
		Ray shadow_ray = Ray(origin,dir_to_light);

		Hit shadow_hit;
		shadow_hit.t = FLT_MAX;

		bool shadow_hit_b = TraceShadow(shadow_ray, shadow_hit, 0.001f);
		if (!shadow_hit_b) {
			answer += (di + si);
		}
	}
	return answer;
}

vec3 GetSkyboxColor(vec3 p) {
	vec2 size = vec2(textureSize(skyboxTex,0));
	float width = size.x / 4.0;
	float height = size.y / 3.0;

    
	if (abs(p.z - skybox.vmin.z) < 0.001f) {
		float offset_width = 3.0 * width;
		float offset_height = 1.0 * height;
		float x_dim = clamp(abs(p.x - skybox.vmin.x) / abs(skybox.vmax.x - skybox.vmin.x), 0.0f, 0.9999f);
		float y_dim = clamp(1.0f - abs((p.y - skybox.vmin.y)) / abs(skybox.vmax.y - skybox.vmin.y), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + x_dim * width, offset_height + y_dim * height);
		vec3 res = texture(skyboxTex,pos/size).rgb;
		return res;
	}
	else if (abs(p.z - skybox.vmax.z) < 0.001f) {
		float offset_width = 1.0 * width;
		float offset_height = 1.0 * height;
		float x_dim = clamp(1.0f - abs(p.x - skybox.vmin.x) / abs(skybox.vmax.x - skybox.vmin.x), 0.0f, 0.9999f);
		float y_dim = clamp(1.0f - abs((p.y - skybox.vmin.y)) / abs(skybox.vmax.y - skybox.vmin.y), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + x_dim * width, offset_height + y_dim * height);
        vec3 res = texture(skyboxTex, pos/size).rgb;
		return res;
	}

	else if (abs(p.x - skybox.vmin.x) < 0.001f) {
		float offset_width = 2.0 * width;
		float offset_height = 1.0 * height;
		float z_dim = clamp(1.0f - abs(p.z - skybox.vmin.z) / abs(skybox.vmax.z - skybox.vmin.z), 0.0f, 0.9999f);
		float y_dim = clamp(1.0f - abs((p.y - skybox.vmin.y)) / abs(skybox.vmax.y - skybox.vmin.y), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + z_dim * width, offset_height + y_dim * height);
        vec3 res = texture(skyboxTex, pos/size).rgb;
		return res;
	}

	else if (abs(p.x - skybox.vmax.x) < 0.001f) {
		float offset_width = 0.0 * width;
		float offset_height = 1.0 * height;
		float z_dim = clamp(abs(p.z - skybox.vmin.z) / abs(skybox.vmax.z - skybox.vmin.z), 0.0f, 0.9999f);
		float y_dim = clamp(1.0f - abs((p.y - skybox.vmin.y)) / abs(skybox.vmax.y - skybox.vmin.y), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + z_dim * width, offset_height + y_dim * height);
        vec3 res = texture(skyboxTex, pos/size).rgb;
		return res;
	}
	else if (abs(p.y - skybox.vmin.y) < 0.001f) {
		float offset_width = 1.0 * width;
		float offset_height = 2.0 * height;
		float x_dim = clamp(1.0f - abs(p.x - skybox.vmin.x) / abs(skybox.vmax.x - skybox.vmin.x), 0.0f, 0.9999f);
		float z_dim = clamp(1.0f - abs((p.z - skybox.vmin.z)) / abs(skybox.vmax.z - skybox.vmin.z), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + x_dim * width, offset_height + z_dim * height);
		vec3 res = texture(skyboxTex, pos/size).rgb;
		return res;
	}

	else if (abs(p.y - skybox.vmax.y) < 0.001f) {
		float offset_width = 1.0 * width;
		float offset_height = 0.0 * height;
		float x_dim = clamp(1.0f - abs(p.x - skybox.vmin.x) / abs(skybox.vmax.x - skybox.vmin.x), 0.0f, 0.9999f);
		float z_dim = clamp(abs((p.z - skybox.vmin.z)) / abs(skybox.vmax.z - skybox.vmin.z), 0.0f, 0.9999f);
		vec2 pos = vec2(offset_width + x_dim * width, offset_height + z_dim * height);
        vec3 res = texture(skyboxTex, pos/size).rgb;
		return res;
	}
	
	return vec3(1.0f, 1.0f, 0.0f);
}

float solveForX(Ray r, float x) {
    return (x - r.origin.x)/r.direction.x; 
}

vec2 rayToXZ(Ray r) {
	return normalize(vec2(r.direction.x,r.direction.z));
}

//Texture space intersection
bool TerrainIntersection(Ray r, float tmin, inout Hit h, inout Material mat) {
    float t;
	float lastT = 0.0;
	vec2 mag = vec2(0);
	float gridDivTwo = gridSize*0.5;
	
	vec2 texSpaceDir = rayToXZ(r);
	vec2 texelSize = vec2(0.5) / vec2(textureSize(heightTex, 0));
	vec2 texSpacePosition = r.origin.xz;
	float maxIte = max(1.0 / texelSize.x, 1.0 / texelSize.y);
	vec2 maxPoint = texSpacePosition + maxIte * texelSize * texSpaceDir;
    
	int ite = 0;
    while(ite < int(maxIte)) {
		vec2 texCoords = ((texSpacePosition)/gridSize) * 0.5 + 0.5;
		texCoords += mag * texSpaceDir;
   
		if(texCoords.x > 1.0 || texCoords.x < 0.0 || texCoords.y > 1.0 || texCoords.y < 0.0) {
			return false;
		}

		float sampledHeight = heightScale * texture(heightTex, texCoords).r - heightScale/2.0;
		vec2 texCoordInWorld = (texCoords * 2.0 - 1.0) * gridSize;
		//Note F(x,z) = y, x and z are on a line, so there is a direct mapping between x -> z
		t = solveForX(r, texCoordInWorld.x);
		vec3 p = PointAtParameter(r,t);

        if(p.y < sampledHeight) {   
			for(int i = 0; i < 8; i++) {
				float tMid = 0.5*(lastT + t);
				p = PointAtParameter(r,tMid);

				if( (p.y-sampledHeight)  < -ACCURACY_THRESH) {
					t = tMid;
				}
				else if( (p.y-sampledHeight)  > ACCURACY_THRESH){
					lastT = tMid;
				}
				else{
					t = tMid;
					break;
				}
			}   
			if(t < tmin) {
				return false;
			}
            h.t = t;
            vec3 normal = texture(normalTex, texCoords).xzy;
            normal = normalize(normal * 2.0f - 1.0f);
            normal.z *= -1.0f;
            h.normal = normal;
            h.material = 0;

            mat.diffuse_color = texture(diffuseTex, texCoords).rgb;
            mat.specular_color = texture(diffuseTex, texCoords).rgb;
            mat.reflective_color = vec3(0.0,0.0,0.0);
            mat.transparent_color = vec3(0.0,0.0,0.0);
            mat.params.y = 8.0;
            mat.params.z = 1.0;
            return true;
        }
		ite++;
		mag += texelSize;
		lastT = t;

    }
    return false;
}

vec2 GetSamplePosition(int n) {
	int m_dim = int(sqrt(float(samples)));
	float incr = 1.0f / float(m_dim);
	vec2 ret = vec2((float((n % m_dim)) * incr + incr * 0.5f), (float((n / m_dim)) * incr + incr * 0.5f));
	return ret;
}

vec2 NormalizedPixelCoordinate(float x, float y) {
	return vec2((x - iResolution.x * 0.5f) / (0.5f * iResolution.x), (-y + iResolution.y * 0.5f) / (0.5f * iResolution.y));
}

Ray GenerateRay(vec2 pixelCoords) {
	Ray r;
	r.origin = camera_center;
	float d = 1.f / tan(camera_fov * 0.5f);
	r.direction = normalize(camera_direction * d + (iResolution.y/iResolution.x) * pixelCoords.y * camera_up + pixelCoords.x * camera_horizontal);
	return r;
}

vec3 RayTrace(Ray r, float tmin, int bounces, Hit hit) {
	vec3 answer = vec3(0.f,0.f,0.f);
	vec3 ref_col = vec3(1.0f, 1.0f, 1.0f);
	vec3 trans_col = vec3(1.0f, 1.0f, 1.0f);
	vec3 fog = vec3(0.7, 0.7, 0.7);
	const float maxDist = 45.0;

	while (bounces >= 0) {
		Material terrainMaterial;
		terrainMaterial.params.z = -1.0;

		bool intersection = false;

		//Sphere
		Hit hT;
		hT.t = FLT_MAX;
		bool bT = TerrainIntersection(r, tmin, hT, terrainMaterial);
		if(bT && hT.t < hit.t) {
			hit = hT;
			intersection = true;
		}	

		if (intersection) {
            Material mat;
            if(terrainMaterial.params.z > 0.0) {
				mat = terrainMaterial;
			} 
			else {
				mat = materials[hit.material];
			}
	
			answer +=  mix(ref_col * Shade(hit, r, mat), fog,  clamp(pow(hit.t,2.0) / pow(maxDist,2.0), 0.0, 1.0));
			ref_col *= mat.reflective_color;

			if (length(ref_col) > 0.0) {
				vec3 ri = normalize(r.direction - 2.f * dot(r.direction, hit.normal) * hit.normal);
				Ray r2 = Ray(PointAtParameter(r, hit.t) + 0.00001f * hit.normal, ri);
				r = r2;
				hit.t = FLT_MAX;
			}
			
			else {
				break;
			}
			
		}
		else {
			if (has_skybox == 1) {
				Hit skybox_hit;
				skybox_hit.t = FLT_MAX;
				bool found = BoxIntersection(skybox, r, tmin, skybox_hit);
				if (found) {
					answer += ref_col * GetSkyboxColor(PointAtParameter(r, skybox_hit.t));
					answer = mix(answer, fog, clamp(pow(hit.t,2.0) / pow(maxDist,2.0), 0.0, 1.0));
					return answer;
				}
			}
			answer += ref_col * scene_background_color;
			answer = mix(answer, fog,  clamp(pow(hit.t,2.0) / pow(maxDist,2.0), 0.0, 1.0));
			return answer;
		}
		bounces--;
	}
	return answer;
}

out vec4 fragColor;

void main()
{
	vec2 pos = vUv * iResolution;
	
    if (pos.x < iResolution.x && pos.y < iResolution.y) {
        skybox = Box(skyboxIn.vmin_s.xyz, skyboxIn.vmax_s.xyz, skyboxIn.params_i_s.xyz);
		int sample_c = int(pow(sqrt(float(samples)), 2.0));
		vec3 out_col = vec3(0.f, 0.0f, 0.0f);
		for (int i = 0; i < sample_c; i++) {
			vec2 offset = GetSamplePosition(i);
			vec2 sample_screen = vec2(pos.x, pos.y) + offset;
			vec2 sample_pos = NormalizedPixelCoordinate(sample_screen.x, sample_screen.y);
			Ray r = GenerateRay(sample_pos);
			Hit h;
			h.t = FLT_MAX;
			out_col += RayTrace(r, camera_tmin, 2, h) / float(sample_c);
		}
		fragColor = vec4(out_col, 1.0f);
        fragColor = vec4(texture(htTex, vUv).rgb, 1.0f);

        return;
	}
    else{
        fragColor = vec4(vUv.x, vUv.y,0,1.0);
    }
}
`;

const hktComputeShader = `

#include <common>

uniform sampler2D H0;
uniform sampler2D H0Conj;

uniform vec2 windowSize;
uniform vec2 waterL;
uniform float t;
uniform float tCycle;

//const float PI = 3.1415;
//const float G = 9.81;

float W(float k) {
	float w0 = 2.0 * 3.1415 / tCycle;
	float w = sqrt(9.81 * k);
	return float(int(w / w0)) * w0;
}

struct Complex {
	float real;
	float imag;
};

Complex convertExp(float exponent) {
	Complex ret;
	ret.real = cos(exponent);
	ret.imag = sin(exponent);
	return ret;
}

Complex add(Complex a, Complex b) {
	Complex ret;
	ret.real = a.real + b.real;
	ret.imag = a.imag + b.imag;
	return ret;
}

Complex mul(Complex a, Complex b) {
	Complex ret;
	ret.real = a.real * b.real - a.imag * b.imag;
	ret.imag = a.real * b.imag + a.imag * b.real;
	return ret;
}

Complex conjugate(Complex a) {
	Complex ret;
	ret.real = a.real;
	ret.imag = a.imag;
	return ret;
}

float alias(float x, float N)
{
	if (x > N / 2.0) {
		x -= N;
    }
	return x;
}

void main()	{

    vec2 cellSize = 1.0 / windowSize.xy;

    vec2 texCoord = gl_FragCoord.xy * cellSize;

    float n = alias(gl_FragCoord.x, float(windowSize.x));
    float m = alias(gl_FragCoord.y,float(windowSize.y));
    
    vec2 k = 2.0 * 3.1415 * vec2(n / waterL.x, m / waterL.y);

    Complex  H0Complex, H0ConjComplex;

    vec2 H0Read = texture(H0, texCoord).rg;
    
    H0Complex.real = H0Read.r;
    H0Complex.imag = H0Read.g;

    vec2 H0ConjRead = texture(H0Conj, texCoord).rg;
    H0ConjComplex.real = H0ConjRead.r;
    H0ConjComplex.imag = -H0ConjRead.g;
    float kMag = max(0.00001, length(k));

    float exponent = W(kMag) * t;

    Complex expH0 = convertExp(exponent);

    Complex expH0Conj = convertExp(-exponent);

    H0Complex = mul(H0Complex, expH0);
    H0ConjComplex = mul(H0ConjComplex, expH0Conj);
    H0ConjComplex.imag = H0ConjComplex.imag;

    Complex complexOut = add(H0Complex, H0ConjComplex);

    gl_FragColor = vec4(complexOut.real, complexOut.imag, 0.0, 1.0);

}
`;

//https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function phillips(A, k, wind_dir) {
    const kAmplitude = k.length()
    const windAmplitude = wind_dir.length()
    const g = 9.81
    const smallWaveFactor = 0.04

    const L = Math.pow(windAmplitude, 2) / g

    const kAmplitudeSq = kAmplitude * kAmplitude
    const smallWaveDampFactor = Math.exp(-kAmplitudeSq * Math.pow(smallWaveFactor, 2))

    const w = wind_dir.clone().normalize()
    const k_dir = k.clone().normalize()

    if (kAmplitude < 0.00001) {
        return 0.0
    }
    const directionCoef = Math.pow((w.dot(k_dir)), 2) * smallWaveDampFactor
    const denominator = 1.0 / Math.pow(kAmplitude, 4.0)
    const waveHeightCoef = A * Math.exp(-1.0 / (Math.pow(kAmplitude * L, 2.0)))
    const ret = waveHeightCoef * directionCoef * denominator
    return ret
}

function alias(x, N)
{
    if (x > N / 2) {
        x -= N
    }
    return x
}

const isqrt2 = 1.0 / Math.sqrt(2.0)
const waterN = 256
const waterM = 256
const waterA = 4.0
var waterL = new THREE.Vector2(1000.0, 1000.0)
var waterWindDir = new THREE.Vector2(21, 0)
var H0 = []
var H0Conj = []

for (var j = 0; j < waterN; j++) {
    for (var i = 0; i < waterM; i++) {
        var n = alias(i, waterN)
        var m = alias(j, waterM)
        var k = new THREE.Vector2(n / waterL.x, m / waterL.y)
        k = (k).multiplyScalar(2.0 * Math.PI)    
        var p = phillips(waterA, k, waterWindDir)
        var e_r = isqrt2 * randn_bm() * Math.sqrt(p)
        var e_i = isqrt2 * randn_bm() * Math.sqrt(p)
        H0.push(e_r)
        H0.push(e_i)
        p = phillips(waterA, k.multiplyScalar(-1), waterWindDir)
        e_r = isqrt2 * randn_bm() * Math.sqrt(p)
        e_i = isqrt2 * randn_bm() * Math.sqrt(p)
        H0Conj.push(e_r)
        H0Conj.push(e_i)
    }
}

const size = waterN * waterM;
const windowSize = new THREE.Vector2(waterN, waterM);

const data1 = new Float32Array(4 * size);
const data2 = new Float32Array(4 * size);
const data3 = new Float32Array(4 * size);

var index = 0;
for ( let i = 0; i < size; i = i + 2 ) {
    data1[index] = H0[i]
    data1[index + 1] = H0[i + 1]
    data1[index + 2] = 0;
    data1[index + 3] = 0;

    data2[index] = H0Conj[i]
    data2[index + 1] = H0Conj[i + 1]
    data2[index + 2] = 0
    data2[index + 3] = 0

    data3[index]  = 0
    data3[index + 1]  = 0
    data3[index + 2]  = 0
    data3[index + 3]  = 0

    index = index + 4;
}

function fillTexture( texture, waterN, waterM, data) {
    const pixels = texture.image.data;

    let p = 0;
    for ( let j = 0; j < 4 * (waterN * waterM); j ++ ) {
        pixels[ j ] = data[j]
    }
 
}

var gpuCompute = new GPUComputationRenderer( waterN, waterM, renderer );

const H0Tex = gpuCompute.createTexture();
const H0ConjTex = gpuCompute.createTexture();
const HtTex = gpuCompute.createTexture();

fillTexture(H0Tex, waterN, waterM, data1);
fillTexture(H0ConjTex, waterN, waterM, data2);
fillTexture(HtTex, waterN, waterM, data3);

const hktVariable = gpuCompute.addVariable( "hkt", hktComputeShader, HtTex );

gpuCompute.setVariableDependencies( hktVariable, [ hktVariable ] );

hktVariable.material.uniforms[ "windowSize" ] = { value: windowSize };
hktVariable.material.uniforms[ "waterL" ] = { value: waterL };
hktVariable.material.uniforms[ "H0" ] = { value: H0Tex };
hktVariable.material.uniforms[ "H0Conj" ] = { value: H0ConjTex };
hktVariable.material.uniforms[ "t" ] = { value: 0 };
hktVariable.material.uniforms[ "tCycle" ] = { value: 20.0 };

const error = gpuCompute.init();
if ( error !== null ) {
    console.error( error );
}

//Variables used in rendering
var cameraDir = new THREE.Vector3();
var cameraPos = new THREE.Vector3();
camera.getWorldDirection(cameraDir)
camera.getWorldPosition(cameraPos)
var cameraUp = camera.up
var cameraHorizontal = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)
cameraHorizontal.cross(cameraUp)
cameraHorizontal.normalize()

var spheresArray = []
var materialsArray = []
var lightsArray = []

const light_1 = {
    position: new THREE.Vector3(0 ,0, 5),
    direction: new THREE.Vector3( 1, -1, 0 ),
    intensity: new THREE.Vector3( 0.8, 0.8, 0.8 ),
    constant_attenuation: new THREE.Vector3( 0.5, 0.5, 0.5 ),
    linear_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
    quadratic_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
    params : new THREE.Vector3( 0, 1, 1),
}

lightsArray.push(light_1)

const sp_1 = {
	center : camera.position.add(cameraDir.clone().multiplyScalar(3)),
    params : new THREE.Vector3(1, 0, 0),
    params_i : new THREE.Vector3(0, -1, 0),
}
spheresArray.push(sp_1);

const material_1 = {
    diffuse_color : new THREE.Vector3(0.0, 0, 0),
    specular_color : new THREE.Vector3(0, 0.0, 0),
    reflective_color : new THREE.Vector3(1.0,1.0,1.0),
    transparent_color : new THREE.Vector3(0.0, 0.0, 0.0),
    params : new THREE.Vector3(0, 4, 0),
}

materialsArray.push(material_1)

//Shader uniforms
const uniforms = {

	skyboxTex : {type: 't', value: null},
    heightTex : {type: 't', value: null},
	normalTex : {type: 't', value: null},
	diffuseTex : {type: 't', value: null},
	htTex : {type: 't', value: null},

    iResolution:  { value: new THREE.Vector2() },
    
    lights : { 
        value: lightsArray
    },

    spheres : { 
        value: spheresArray
    },
    
    skyboxIn : { 
        value: {
			vmin_s : new THREE.Vector3(-30, -30, -30),
			vmax_s : new THREE.Vector3(30, 30, 30),
			params_i_s : new THREE.Vector3(0, -1, 0),
		}
    },
    
    materials : { 
        value: materialsArray  
    },

    scene_ambient_light : {
        value : new THREE.Vector3(0.1, 0.1, 0.1),
    },

    scene_background_color : {
        value : new THREE.Vector3(0.1, 0.1, 0.1)
    },

    camera_center : {
        value : cameraPos
    },

    camera_direction : {
        value : cameraDir
    },

    camera_up : {
        value : camera.up
    },

    camera_horizontal : {
        value : cameraHorizontal
    },
    
    camera_fov : {
        value : 75.0/180.0 * Math.PI
    },

    camera_tmin : {
        value : 0.01
    },

    light_count : {
        value : lightsArray.length
    },
    
    sphere_count : {
        value : spheresArray.length
    },

    has_skybox : {
        value : 1
    },

    samples : {
        value : 2
    },

    heightScale : {
        value: terrainHeightScale

    },

    gridSize : {
        value: 30
    },
};

const skyboxTexture = new THREE.TextureLoader().load( 
	"images/skyboxDay.png",
);

const heightTexture = new THREE.TextureLoader().load( 
	"images/texturingHeight.png",
);

const normalTexture = new THREE.TextureLoader().load( 
	"images/texturingNormal.png",
);

const diffuseTexture = new THREE.TextureLoader().load( 
	"images/texturingDiffuse.png",
);

skyboxTexture.wrapS = THREE.ClampToEdgeWrapping
skyboxTexture.wrapT = THREE.ClampToEdgeWrapping
skyboxTexture.magFilter = THREE.NearestFilter
skyboxTexture.minFilter = THREE.NearestFilter
skyboxTexture.repeat.set( 4, 4 )

heightTexture.wrapS = THREE.ClampToEdgeWrapping
heightTexture.wrapT = THREE.ClampToEdgeWrapping
heightTexture.magFilter = THREE.LinearFilter
heightTexture.minFilter = THREE.LinearFilter
heightTexture.repeat.set( 4, 4 )

normalTexture.wrapS = THREE.ClampToEdgeWrapping
normalTexture.wrapT = THREE.ClampToEdgeWrapping
normalTexture.magFilter = THREE.LinearFilter
normalTexture.minFilter = THREE.LinearFilter
normalTexture.repeat.set( 4, 4 )

diffuseTexture.wrapS = THREE.ClampToEdgeWrapping
diffuseTexture.wrapT = THREE.ClampToEdgeWrapping
diffuseTexture.magFilter = THREE.LinearFilter
diffuseTexture.minFilter = THREE.LinearFilter
diffuseTexture.repeat.set( 4, 4 )

uniforms.skyboxTex.value = skyboxTexture
uniforms.heightTex.value = heightTexture
uniforms.normalTex.value = normalTexture
uniforms.diffuseTex.value = diffuseTexture
uniforms.htTex.value = HtTex;

//Fullscreen quad for rendering
var quad = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms,
        glslVersion: THREE.GLSL3,
        depthWrite: false,
        depthTest: false,
    })
  );
scene.add(quad)

var cameraLookAtVector = new THREE.Vector3(0,-5,10)
var rotAngle = Math.PI - Math.PI/4;
const rotSpeed = 0.1
var rotMat = new THREE.Matrix4()

function animate(time) {
	var delta = clock.getDelta()
	rotAngle += (delta * rotSpeed) % (2.0 * Math.PI);
	rotMat = rotMat.makeRotationY(rotAngle)
	var dir = cameraLookAtVector.clone().applyMatrix4(rotMat)
	cameraPos = camera.getWorldPosition(cameraPos)

	var currentLookAtVector = cameraPos.clone().add(dir)
	controls.target = currentLookAtVector

	requestAnimationFrame(animate)
    controls.update(delta)
	stats.update()
	render(delta)
}

var tCumul = 0.0;
function render(time) {
    tCumul += time;
    //Compute 
    hktVariable.material.uniforms["t"].value  = tCumul;
    gpuCompute.compute();

    const canvas = renderer.domElement
    uniforms.iResolution.value.set(canvas.width, canvas.height)

	//Update camera direction etc.
    cameraDir = new THREE.Vector3()
    cameraPos  = new THREE.Vector3()
    cameraDir = camera.getWorldDirection(cameraDir)

    cameraPos = camera.getWorldPosition(cameraPos)
	cameraHorizontal = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)
	cameraHorizontal.cross(camera.up)
    cameraHorizontal.normalize();
	var cameraUp = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)
	cameraUp.cross(cameraHorizontal)
	cameraUp.normalize();

	uniforms.camera_direction.value = cameraDir
    uniforms.camera_horizontal.value = cameraHorizontal
    uniforms.camera_center.value = cameraPos
	uniforms.camera_up.value = cameraUp
    uniforms.htTex.value = gpuCompute.getCurrentRenderTarget( hktVariable ).texture;

    renderer.render(scene, renderCamera)
}

animate()


