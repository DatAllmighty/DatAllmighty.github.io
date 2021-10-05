import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls'
import Stats from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/stats.module'
import { GPUComputationRenderer } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/misc/GPUComputationRenderer.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/dat.gui.module'

//Parent div
const parent = document.getElementById("embedded-video-div")

//Camera Parameters
let cameraDir = new THREE.Vector3();
let cameraPos = new THREE.Vector3();
let cameraUp = new THREE.Vector3();
let cameraHorizontal = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)

let cameraLookAtVector = new THREE.Vector3(-2.6,-0.6,-19.4)
let rotAngle = 0.0
let rotSpeed = 0.0
let rotMat = new THREE.Matrix4()

//Terrain parameters
var terrainHeightScale = 7
let skyboxTexture, heightTexture, normalTexture, diffuseTexture

//Water generation parameters
const waterHeightScale = 10
const isqrt2 = 1.0 / Math.sqrt(2.0)
const waterN = 256
const waterM = 256
let waterA = 800
let waterHeightOffset = 7
let waterGridSize = 30

const waterL = new THREE.Vector2(64, 64);
var waterWindDir = new THREE.Vector2(2.5, 0)
const size = waterN * waterM;
const windowSize = new THREE.Vector2(waterN, waterM);
const data1 = new Float32Array(4 * size);
const data2 = new Float32Array(4 * size);
const data3 = new Float32Array(4 * size);
let H0 = []
let H0Conj = []
let H0Tex 
let H0ConjTex
let HtTex 
let hktVariable
let idftComputeShader, normalComputeShader
let horisontalSumTex, waterHeightTex, waterNormalTexture
let waterUpdateInterval = 1.0/30.0
let lastWaterUpdateTime = 0
let enableRefraction = false;

//Other scene & control parameters
let camera, renderCamera, renderer, stats, scene, quad, controls, clock, gpuCompute, totalTime = 0.0, uniforms;
let gui, guiOptions, waterFolder, waveFolder, cameraFolder;

const materialsArray = []
const lightsArray = []

function initRendering() {
    //Create rendering context
    renderer = new THREE.WebGLRenderer()
    renderer.outputEncoding = THREE.sRGBEncoding
    renderer.setSize(parent.clientWidth, parent.clientHeight)
    parent.appendChild(renderer.domElement)
    stats = Stats()
    parent.appendChild(stats.dom)

    //Create camera & controls, "forced" orthographic render camera for screenspace quad
    camera = new THREE.PerspectiveCamera(
        75,
        parent.clientWidth / parent.clientHeight,
        0.01,
        1000
    )

    camera.position.x = -5.25
    camera.position.y = 4.7
    camera.position.z = 11

    renderCamera = new THREE.OrthographicCamera( -1, 1, 1, -1, 0, 1 )
    controls = new OrbitControls( camera, parent )
    controls.enabled = false;
    clock = new THREE.Clock();

    //Set resize callbacks
    window.addEventListener('resize', onWindowResize, false)
    function onWindowResize() {
        camera.aspect = parent.clientWidth / parent.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(parent.clientWidth, parent.clientHeight)
}

//Create scene
scene = new THREE.Scene()
scene.add(new THREE.AxesHelper(5))
}

//Shaders
const vertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
    }
`
const fragmentShader = `

#define PI 3.141592
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
	//Fresnel, roughness, metalness
    vec3 params;
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

uniform Material materials[1];

uniform Light lights[1];

uniform SkyboxParams skyboxIn;

uniform vec3 scene_ambient_light;
uniform vec3 scene_background_color;
uniform vec3 camera_center;
uniform vec3 camera_direction;
uniform vec3 camera_up;
uniform vec3 camera_horizontal;

uniform vec2 windowSize;

uniform float camera_fov;
uniform float camera_tmin;

uniform int light_count;
uniform int	has_skybox;

uniform int	samples;

uniform sampler2D skyboxTex;
uniform sampler2D heightTex;
uniform sampler2D normalTex;
uniform sampler2D diffuseTex;
uniform sampler2D waterHeightTex;
uniform sampler2D waterNormalTex;

uniform float heightScale;
uniform float waterHeightScale;
uniform float gridSize;
uniform float waterGridSize;
uniform float waterHeightOffset;

uniform bool enableRefraction;

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


//https://learnopengl.com/PBR/Theory
float DistributionGGX(vec3 N, vec3 H, float a)
{
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;
	
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float k)
{
    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}
  
float GeometrySmith(vec3 N, vec3 V, vec3 L, float k)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);
	
    return ggx1 * ggx2;
}

float FresnelSchlick(vec3 L, vec3 H, float ni) {
    const float n1 = 1.0;
    float n2 = ni;
    float n = n1 / n2;
    float dot = dot(L, H);
    float n_s = pow(n, 2.0);
    float B = sqrt((1.0 / n_s) + pow(dot, 2.0) - 1.0);
    float Rs = pow((dot - B) / (dot + B), 2.0);
    float Rp = pow((n_s * B - dot) / (n_s * B + dot), 2.0);
    return 0.5 * (Rs + Rp);
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

float solveForX(Ray r, float x) {
    return (x - r.origin.x)/r.direction.x; 
}

vec2 rayToXZ(Ray r) {
	return normalize(vec2(r.direction.x,r.direction.z));
}

//Texture space intersection
bool TerrainIntersection(Ray r, float tmin, inout Hit h, inout Material mat, float step) {
    float t;
	float lastT = 0.0;
	vec2 mag = vec2(0);
	
	vec2 texSpaceDir = rayToXZ(r);
	vec2 texelSize = vec2(step) / vec2(textureSize(heightTex, 0));
	vec2 texSpacePosition = r.origin.xz;
	float maxIte = max(1.0 / texelSize.x, 1.0 / texelSize.y);
    
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
			for(int i = 0; i < 4; i++) {
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
            mat.params.x = 1.51;
            mat.params.y = 1.0;
            mat.params.z = 0.0;
            return true;
        }
		ite++;
		mag += texelSize;
		lastT = t;

    }
    return false;
}

bool WaterIntersection(Ray r, float tmin, inout Hit h, inout Material mat) {
    float t;
	float lastT = 0.0;
	vec2 mag = vec2(0);
	
	vec2 texSpaceDir = rayToXZ(r);
	vec2 texelSize = vec2(0.5) / vec2(textureSize(waterHeightTex, 0));
	vec2 texSpacePosition = r.origin.xz;
	float maxIte = max(1.0 / texelSize.x, 1.0 / texelSize.y);
    
	int ite = 0;
    while(ite < int(maxIte)) {
		vec2 texCoords = ((texSpacePosition)/waterGridSize) * 0.5 + 0.5;
		texCoords += mag * texSpaceDir;
   
		if(texCoords.x > 1.0 || texCoords.x < 0.0 || texCoords.y > 1.0 || texCoords.y < 0.0) {
			return false;
		}

		float sampledHeight = waterHeightScale * texture(waterHeightTex, texCoords).r - waterHeightScale/2.0 + waterHeightOffset;
		vec2 texCoordInWorld = (texCoords * 2.0 - 1.0) * waterGridSize;
		//Note F(x,z) = y, x and z are on a line, so there is a direct mapping between x -> z
		t = solveForX(r, texCoordInWorld.x);
		vec3 p = PointAtParameter(r,t);

        if(t > h.t){
            return false;
        }        
        if(p.y < sampledHeight) {   
			for(int i = 0; i < 4; i++) {
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
            vec3 normal = texture(waterNormalTex, texCoords).xyz;
            h.normal = normalize(normal);
            h.material = 0;

            mat.diffuse_color = vec3(6.0/255.0,66.0/255.0,115.0/255.0);
            mat.specular_color = vec3(0.0);
            mat.reflective_color = vec3(0.1,0.1,0.1);
            mat.transparent_color = vec3(0.0,0.0,0.0);
            mat.params.x = 1.33;
            mat.params.y = 0.2;
            mat.params.z = 0.0;
            return true;
        }
		ite++;
		mag += texelSize;
		lastT = t;

    }
    return false;
}

bool TraceShadow(Ray r, Hit h, float tmin) {
	bool intersection = false;

    /*
    Material terrainMaterial;

    if(TerrainIntersection(r, tmin, h, terrainMaterial)) {
        return true;
    }
	*/
    return intersection;
}


vec3 Shade(Hit hit, Ray r, Material mat) {
	
	vec3 answer = mat.diffuse_color * scene_ambient_light;
    vec3 N = hit.normal;
    vec3 V = -r.direction;

    for (int i = 0; i < light_count; i++) {
		Light light = lights[i];
		vec3 I;
		vec3 L;
		float distance;
		GetIncidentIllumination(light, r, hit, I, L, distance);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);        

        float F    = FresnelSchlick(L, H, mat.params.x);
        float NDF = DistributionGGX(N, H, mat.params.y);       
        float G   = GeometrySmith(N, V, L, mat.params.y);   
            
        float numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL;
        float spec = numerator / (denominator + 1.0);
        vec3 specular     = vec3(spec);  
       
        float kS = F;
        float kD = 1.0 - kS;
        kD *= 1.0 - mat.params.z;	
        
        kD = 1.0;
		vec3 origin = PointAtParameter(r, hit.t) + 0.1f * N;
		Ray shadowRay = Ray(origin, L);
        
		Hit shadowHit;
		shadowHit.t = FLT_MAX;
        
		bool shadowHitB = TraceShadow(shadowRay, shadowHit, 0.01f);
       
		if (!shadowHitB) {
			answer += ( kD * mat.diffuse_color + specular ) * I * NdotL;
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

vec2 GetSamplePosition(int n) {
	int m_dim = int(sqrt(float(samples)));
	float incr = 1.0f / float(m_dim);
	vec2 ret = vec2((float((n % m_dim)) * incr + incr * 0.5f), (float((n / m_dim)) * incr + incr * 0.5f));
	return ret;
}

vec2 NormalizedPixelCoordinate(float x, float y) {
	return vec2((x - windowSize.x * 0.5f) / (0.5f * windowSize.x), (-y + windowSize.y * 0.5f) / (0.5f * windowSize.y));
}

Ray GenerateRay(vec2 pixelCoords) {
	Ray r;
	r.origin = camera_center;
	float d = 1.f / tan(camera_fov * 0.5f);
	r.direction = normalize(camera_direction * d + (windowSize.y/windowSize.x) * pixelCoords.y * camera_up + pixelCoords.x * camera_horizontal);
	return r;
}

vec3 RayTrace(Ray r, float tmin, int bounces, Hit hit) {
	vec3 answer = vec3(0.f,0.f,0.f);
	vec3 ref_col = vec3(1.0f, 1.0f, 1.0f);
	vec3 trans_col = vec3(1.0f, 1.0f, 1.0f);
	const float maxDist = 45.0;
    vec3 water_depth_col = vec3(6.0/255.0,66.0/255.0,115.0/255.0);
    float maxVisibleDepth = 1.0f;

	while (bounces >= 0) {
		Material terrainMaterial, waterMaterial, refractionMaterial;
		terrainMaterial.params.z = -1.0;
		waterMaterial.params.z = -1.0;
		bool intersection = false;
          
        Hit hW;
		hW.t = FLT_MAX;
		bool bW = WaterIntersection(r, tmin, hW, waterMaterial);
		if(bW && hW.t < hit.t) {
			hit = hW;
			intersection = true;
		}	
        
        Ray terrainR = r;
        bool refraction = false;
        vec3 waterP;

        if(intersection) {
            waterP = PointAtParameter(r, hit.t);
            //Fetch height of terrain at the intersected point. If it is higher -> we trace water refraction otherwise normal
            vec2 texSpacePosition = waterP.xz;
            vec2 texCoords = ((texSpacePosition)/gridSize) * 0.5 + 0.5;
            float sampledHeight = heightScale * texture(heightTex, texCoords).r - heightScale/2.0;
            if(sampledHeight < waterP.y) {
                refraction = true && enableRefraction;
            }
        }

        Hit hT;
        hT.t = FLT_MAX;

        if(refraction) {
            hT.t = tmin;
            float n = 1.0 / waterMaterial.params.x;
            float cosI = -dot(hit.normal, r.direction);
            float sinT2 = n * n * (1.0 - cosI * cosI);
            float cosT = sqrt(1.0 - sinT2);
            terrainR.origin = waterP;
            terrainR.direction =  n * r.direction + (n * cosI - cosT) * hit.normal;
            bool bT = TerrainIntersection(terrainR, tmin, hT,  refractionMaterial, 1.0f);
            refraction = bT;
        }
        else{
            bool bT = TerrainIntersection(terrainR, tmin, hT, terrainMaterial, 1.0f);
            if(bT && hT.t < hit.t) {
                hit = hT;
                intersection = true;
                waterMaterial.params.z = -1.0;
            }
            else{
                terrainMaterial.params.z = -1.0;
            }
        }
       
    
		if (intersection) {
            Material mat;
            if(terrainMaterial.params.z > -0.1) {
				mat = terrainMaterial;
			} 
            else if(waterMaterial.params.z > -0.1) {
				mat = waterMaterial;
            }
			else {
				mat = materials[hit.material];
			}
	

            if(refraction) {
                vec3 refractedCol = Shade(hT, terrainR,  refractionMaterial);
                answer += mix(refractedCol,ref_col * Shade(hit, r, mat), clamp(hT.t/maxVisibleDepth, 0.0, 1.0));
            }
            else {
                answer +=  ref_col * Shade(hit, r, mat);
            }

			ref_col *= mat.reflective_color;
            
			if (length(ref_col) > 0.0) {
				vec3 ri = normalize(r.direction - 2.f * dot(r.direction, hit.normal) * hit.normal);
				Ray r2 = Ray(PointAtParameter(r, hit.t) + 0.001f * hit.normal, ri);
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
					return answer;
				}
			}
			answer += ref_col * scene_background_color;
			return answer;
		}
		bounces--;
	}
	return answer;
}

out vec4 fragColor;

void main()
{
	vec2 pos = vUv * windowSize;

    if (pos.x < windowSize.x && pos.y < windowSize.y) {
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
    float m = alias(gl_FragCoord.y, float(windowSize.y));
    
    vec2 k = 2.0 * 3.1415 * vec2(n, m) / waterL;

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
const idftShader = `

//Created with the help from https://www.shadertoy.com/view/tdSfWG
#include <common>

#define TWO_PI 6.283185

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

Complex mul(Complex a, Complex b) {
	Complex ret;
	ret.real = a.real * b.real - a.imag * b.imag;
	ret.imag = a.real * b.imag + a.imag * b.real;
	return ret;
}

uniform vec2 windowSize;
uniform vec2 waterL;
uniform sampler2D hkt;
uniform sampler2D horizontalTex;
uniform bool horizontal;

void main() {
    vec4 col = vec4(0);

    if(gl_FragCoord.x < windowSize.x + 1.0 && gl_FragCoord.y < windowSize.y + 1.0){

        float column = gl_FragCoord.x;
        float row = gl_FragCoord.y;

        for(float n = 0.0; n < SIZE; n++){
            Complex h_tilde_complex, exp;
            float m = n;
            vec2 a = TWO_PI * (gl_FragCoord.xy - 0.5) * n/SIZE;
            if(horizontal) {
                vec2 h_tilde = texelFetch(hkt, ivec2(m, row), 0).xy;
                exp = convertExp(a.x);
                h_tilde_complex.real = h_tilde.x;
                h_tilde_complex.imag = h_tilde.y;
                h_tilde_complex = mul(h_tilde_complex, exp);
                col.x += h_tilde_complex.real;
                col.y += h_tilde_complex.imag;

            }
            else{
                vec2 horizontalSum = texelFetch(horizontalTex, ivec2(column, m), 0).xy;
                exp = convertExp(a.y);
                h_tilde_complex.real = horizontalSum.x;
                h_tilde_complex.imag = horizontalSum.y;
                h_tilde_complex = mul(h_tilde_complex, exp);
                col.x += h_tilde_complex.real;
                col.y += h_tilde_complex.imag;
            }
        }
        col.xy /= SIZE;
    }
    gl_FragColor = col;
}

`;
const normalShader = `
//Created with the help from https://www.shadertoy.com/view/tdSfWG
uniform vec2 windowSize;
uniform float waterHeightScale;
uniform sampler2D heightMap;
uniform float mPerPixel;
void main() {
    vec3 normal = vec3(0.0);
    if(gl_FragCoord.x < windowSize.x && gl_FragCoord.y < windowSize.y){
        vec2 uv = gl_FragCoord.xy/windowSize;
        vec2 texelSize = 1.0 / vec2(textureSize(heightMap,0).xy);
		vec3 a = vec3(-mPerPixel, waterHeightScale * texture(heightMap, uv + vec2(-texelSize.x,0.0)).r,  0.0);
    	vec3 b = vec3( mPerPixel, waterHeightScale * texture(heightMap, uv + vec2(texelSize.x,0.0)).r,  0.0);
		vec3 c = vec3( 0.0, waterHeightScale * texture(heightMap, uv + vec2(0.0,-texelSize.y)).r, -mPerPixel);
    	vec3 d = vec3( 0.0, waterHeightScale * texture(heightMap, uv + vec2(0.0,texelSize.y)).r,  mPerPixel);
    	normal = normalize(cross(c - d, a - b));
    }
    gl_FragColor = vec4(normal, 1.0);
}

`;

//https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
function randn_bm() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function phillips(A, k, wind_dir) {
    const kAmplitude = k.length()
    const windAmplitude = wind_dir.length()
    const g = 9.81
    const smallWaveFactor = 0.01

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
    const waveHeightCoef = A * Math.exp(-1.0 / (kAmplitude * Math.pow(L, 2.0)))
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


function fillTexture( texture, waterN, waterM, data) {
    const pixels = texture.image.data;
    for ( let j = 0; j < 4 * (waterN * waterM); j ++ ) {
        pixels[ j ] = data[j]
    }
 
}

function setWaterNoiseTextures() {
    H0 = []
    H0Conj = []

    H0Tex = gpuCompute.createTexture();
    H0ConjTex = gpuCompute.createTexture();
    HtTex = gpuCompute.createTexture();

    for (let j = 0; j < waterN; j++) {
        for (let i = 0; i < waterM; i++) {
            let n = alias(i, waterN)
            let m = alias(j, waterM)
            let k = new THREE.Vector2(n / waterL.x, m / waterL.y), k2
            k = (k).multiplyScalar(2.0 * Math.PI)    
            let p = phillips(waterA, k, waterWindDir)
            let e_r = isqrt2 * randn_bm() * Math.sqrt(p)
            let e_i = isqrt2 * randn_bm() * Math.sqrt(p)
            H0.push(e_r)
            H0.push(e_i)
            k2 = k.multiplyScalar(-1);
            p = phillips(waterA, k2, waterWindDir)
            e_r = isqrt2 * randn_bm() * Math.sqrt(p)
            e_i = isqrt2 * randn_bm() * Math.sqrt(p)
            H0Conj.push(e_r)
            H0Conj.push(e_i)
        }
    }
    
    let index = 0;
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

    fillTexture(H0Tex, waterN, waterM, data1);
    fillTexture(H0ConjTex, waterN, waterM, data2);
    fillTexture(HtTex, waterN, waterM, data3);

    hktVariable.material.uniforms[ "H0" ] = { value: H0Tex };
    hktVariable.material.uniforms[ "H0Conj" ] = { value: H0ConjTex };
}


function initComputeShaders() {
    gpuCompute = new GPUComputationRenderer( waterN, waterM, renderer );

    H0Tex = gpuCompute.createTexture();
    H0ConjTex = gpuCompute.createTexture();
    HtTex = gpuCompute.createTexture();
    
    horisontalSumTex = new THREE.WebGLRenderTarget( waterN, waterM, {
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        depthBuffer: false
    } );
    
    waterHeightTex = new THREE.WebGLRenderTarget( waterN, waterM, {
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        depthBuffer: false
    } );
    
    waterNormalTexture = new THREE.WebGLRenderTarget( waterN, waterM, {
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        depthBuffer: false
    } );
    
    hktVariable = gpuCompute.addVariable( "hkt", hktComputeShader, HtTex );
    
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
    
    idftComputeShader = gpuCompute.createShaderMaterial( idftShader, {
        windowSize: { value: new THREE.Vector2(waterN, waterM) },
        horizontalTex: { value: null },
        hkt: { value: null },
        horizontal: { value: false}
    } );
    
    idftComputeShader.defines.SIZE = waterN.toFixed( 1 );
    
    normalComputeShader = gpuCompute.createShaderMaterial( normalShader, {
        windowSize: { value: new THREE.Vector2(waterN, waterM) },
        heightMap: { value: null },
        waterHeightScale: { value: waterHeightScale},
        mPerPixel: { value: waterL.x/waterN}
    } );

}

function runComputeShader(time) {

    hktVariable.material.uniforms["t"].value  = time;
    gpuCompute.compute();

    idftComputeShader.uniforms['horizontal'].value = true
    idftComputeShader.uniforms['hkt'].value = gpuCompute.getCurrentRenderTarget( hktVariable ).texture
    gpuCompute.doRenderTarget(idftComputeShader, horisontalSumTex)
    idftComputeShader.uniforms['hkt'].value = null
    idftComputeShader.uniforms['horizontal'].value = false
    idftComputeShader.uniforms['horizontalTex'].value = horisontalSumTex.texture
    gpuCompute.doRenderTarget(idftComputeShader, waterHeightTex)
    idftComputeShader.uniforms['horizontalTex'].value = null

    normalComputeShader.uniforms['heightMap'].value = waterHeightTex.texture
    gpuCompute.doRenderTarget(normalComputeShader, waterNormalTexture)

}

function initScene() {

    //Create gui to modify water params
    gui = new GUI()
    guiOptions = {
        WaterAmplitude : waterA,
        CameraRotationSpeed : rotSpeed,
        HeightOffset : waterHeightOffset,
        WaterGridSize : waterGridSize,
        TerrainHeightScale : terrainHeightScale,
        EnableRefraction : enableRefraction,
    }
    waterFolder = gui.addFolder('Wind direction')

    waterFolder.add(waterWindDir, 'x', -100, 100).step(0.05).onChange( function ( value ) {                    
            setWaterNoiseTextures()
    } );
    waterFolder.add(waterWindDir, 'y', -100, 100).step(0.05).onChange( function ( value ) {                    
            setWaterNoiseTextures()
    } );
    waterFolder.open()
    waveFolder = gui.addFolder('Water')

    waveFolder.add(guiOptions, 'WaterAmplitude', 0, 2000).step(0.05).onChange( function ( value ) {                    
        waterA = value;
        setWaterNoiseTextures()
    } );

    waveFolder.add(guiOptions, 'HeightOffset', -30, 30).step(0.05).onChange( function ( value ) {                    
        waterHeightOffset = value;
    } );

    waveFolder.add(guiOptions, 'WaterGridSize', -30, 30).step(0.5).onChange( function ( value ) {                    
        waterGridSize = value;
    } );

    
    waveFolder.add(guiOptions, 'EnableRefraction', false, true).onChange( function ( value ) {                    
        enableRefraction = value;
    } );

    waveFolder = gui.addFolder('Terrain')

    waveFolder.add(guiOptions, 'TerrainHeightScale', 1, 20).step(0.05).onChange( function ( value ) {                    
       terrainHeightScale = value;
    } )
    waveFolder.open()

    cameraFolder = gui.addFolder('Camera')
    cameraFolder.add(camera.position, 'x', -30, 30)
    cameraFolder.add(camera.position, 'y', -30, 30)
    cameraFolder.add(camera.position, 'z', -30, 30)
    cameraFolder.add(guiOptions, 'CameraRotationSpeed', -30, 30).step(0.01).onChange( function ( value ) {                    
        rotSpeed = value
    } );
    cameraFolder.open()

    const light_1 = {
        position: new THREE.Vector3(0 ,0, 5),
        direction: new THREE.Vector3( 1, -1, 1 ),
        intensity: new THREE.Vector3( 0.5,  0.5,  0.5 ),
        constant_attenuation: new THREE.Vector3( 0.5, 0.5, 0.5 ),
        linear_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
        quadratic_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
        params : new THREE.Vector3( 0, 1, 1),
    }
    lightsArray.push(light_1)
    
    const material_1 = {
        diffuse_color : new THREE.Vector3(0.0, 0, 0),
        specular_color : new THREE.Vector3(0, 0.0, 0),
        reflective_color : new THREE.Vector3(1.0,1.0,1.0),
        transparent_color : new THREE.Vector3(0.0, 0.0, 0.0),
        params : new THREE.Vector3(0, 4, 0),
    }
    materialsArray.push(material_1)
    
    //Shader uniforms
    uniforms = {
    
        skyboxTex : {type: 't', value: null},
        heightTex : {type: 't', value: null},
        normalTex : {type: 't', value: null},
        diffuseTex : {type: 't', value: null},
        waterHeightTex : {type: 't', value: null},
        waterNormalTex : {type: 't', value: null},
    
        windowSize:  { value: new THREE.Vector2() },
        
        lights : { 
            value: lightsArray
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
            value : new THREE.Vector3(0.0, 0.0, 0.0),
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
           
        has_skybox : {
            value : 1
        },
    
        samples : {
            value : 2
        },
    
        heightScale : {
            value: terrainHeightScale
    
        },
    
        waterHeightScale : {
            value: waterHeightScale
    
        },

        waterHeightOffset : {
            value: waterHeightOffset
    
        },

        gridSize : {
            value: 25
        },
    
        waterGridSize : {
            value: 30
        },

        enableRefraction : {
            value : enableRefraction
        }
    };
    
    skyboxTexture = new THREE.TextureLoader().load( 
        "images/skyboxDay.png",
    );
    
    heightTexture = new THREE.TextureLoader().load( 
        "images/monumentHeight.png",
    );
    
    normalTexture = new THREE.TextureLoader().load( 
        "images/monumentNormal.png",
    );
    
    diffuseTexture = new THREE.TextureLoader().load( 
        "images/monumentDiffuse.png",
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
    
    quad = new THREE.Mesh(
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
}

function updateCamera(delta) {
    //Rotate camera
    rotAngle += (delta * rotSpeed) % (2.0 * Math.PI);
	rotMat = rotMat.makeRotationY(rotAngle)
	let dir = cameraLookAtVector.clone().applyMatrix4(rotMat)
	cameraPos = camera.getWorldPosition(cameraPos)
	let currentLookAtVector = cameraPos.clone().add(dir)
	controls.target = currentLookAtVector

	//Update camera direction etc.
    cameraDir = new THREE.Vector3()
    cameraPos  = new THREE.Vector3()
    cameraDir = camera.getWorldDirection(cameraDir)

    cameraPos = camera.getWorldPosition(cameraPos)
	cameraHorizontal = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)
	cameraHorizontal.cross(camera.up)
    cameraHorizontal.normalize();
	cameraUp = new THREE.Vector3(cameraDir.x, cameraDir.y, cameraDir.z)
	cameraUp.cross(cameraHorizontal)
	cameraUp.normalize();
}

//Set main shader(ray trace) uniforms
function updateUniforms() {
    const canvas = renderer.domElement
    uniforms.windowSize.value.set(canvas.width, canvas.height)
	uniforms.camera_direction.value = cameraDir
    uniforms.camera_horizontal.value = cameraHorizontal
    uniforms.camera_center.value = cameraPos
	uniforms.camera_up.value = cameraUp
    uniforms.heightScale.value = terrainHeightScale;
    uniforms.waterHeightTex.value = waterHeightTex.texture
    uniforms.waterNormalTex.value = waterNormalTexture.texture
    uniforms.waterHeightOffset.value = waterHeightOffset
    uniforms.waterGridSize.value = waterGridSize
    uniforms.enableRefraction.value = enableRefraction

}

function animate(time) {
    let delta = clock.getDelta()
    requestAnimationFrame(animate)
    updateCamera(delta);
    controls.update(delta)
	stats.update()
	render(delta)
}

function render(time) {
    totalTime += time;
    updateUniforms()
    if(totalTime - lastWaterUpdateTime > waterUpdateInterval) {
        runComputeShader(totalTime)
        lastWaterUpdateTime = totalTime;
    }
    //Raytrace
    renderer.render(scene, renderCamera)
}

//Enter render loop
initRendering()
initScene()
initComputeShaders()
setWaterNoiseTextures()
animate()


