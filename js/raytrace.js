import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls'
import Stats from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/stats.module'

var parent = document.getElementById("embedded-video-div")

const renderer = new THREE.WebGLRenderer()
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(parent.clientWidth, parent.clientHeight)
parent.appendChild(renderer.domElement)

const camera = new THREE.PerspectiveCamera(
    75,
    parent.clientWidth / parent.clientHeight,
    0.01,
    1000
)
camera.position.y = 5
camera.position.z = 13
camera.position.x = 0

const scene = new THREE.Scene()

scene.add(new THREE.AxesHelper(5))
const controls = new OrbitControls(camera, renderer.domElement)

const stats = Stats()
parent.appendChild(stats.dom)

window.addEventListener('resize', onWindowResize, false)

function onWindowResize() {
    camera.aspect = parent.clientWidth / parent.clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(parent.clientWidth, parent.clientHeight)
}

const vertexShader = `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
    }
`
const fragmentShader = `
#define FLT_MAX 3.402823466e+38F   
const float coef = 0.5;

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

struct Transform
{
	mat4 transform;
	mat4 transform_inverse;
	mat4 transform_inverse_transpose;
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
//params (offset, null null)
//params_i (material, transform, null)

struct Plane {
	vec3 normal;
	vec3 params;
	vec3 params_i;
};


struct SkyboxParams {
    vec3 vmin_s;
	vec3 vmax_s;
	vec3 params_i_s;
};

uniform Sphere spheres[26];

uniform Sphere blobs[25];

uniform Material materials[26];

uniform	Box boxes[1];

uniform Light lights[1];

uniform Plane planes[1];

uniform SkyboxParams skyboxIn;

uniform vec3 scene_ambient_light;
uniform vec3 scene_background_color;

uniform vec3 camera_center;
uniform vec3 camera_direction;
uniform vec3 camera_up;
uniform vec3 camera_horizontal;
uniform float camera_fov;
uniform float camera_tmin;

uniform int light_count;
uniform int sphere_count;
uniform int blob_count;
uniform int	box_count;
uniform int	plane_count;
uniform int	has_skybox;

uniform int	samples;

uniform vec2 iResolution;
uniform float iTime;

uniform sampler2D skyboxTex;

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

float evalF(vec3 pos, vec3 center, float radius) {
	float r = length(pos - center);
	if (r > radius * 2.0) {
		return 0.0;
	}
	else {
		radius *= 2.0;
		float force = 2.0 * (pow(r, 3.0) * (1.0 / pow(radius, 3.0))) - 3.0 * (pow(r, 2.0) * (1.0 / pow(radius, 2.0))) + 1.0;
		return force;
	}
}

vec3 evalN(vec3 pos, vec3 center, float radius) {
	float r = length(pos - center);
	if (r > radius * 2.0) {
		return vec3(0);
	}
	radius *= 2.0;

	float rec = 1.0 / (pow(pos.x - center.x, 2.0) + pow(pos.y - center.y, 2.0) + pow(pos.z - center.z, 2.0));
	return rec * 2.f * (pos-center);
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

bool SphereIntersectionWithMod(Sphere sp, Ray r, float tmin, inout Hit hit, float mod) {

	vec3 tmp = sp.center - r.origin;
	vec3 dir = r.direction;
	float radius = sp.params.x * mod;
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

bool BlobIntersection(Ray r, float tmin, inout Hit h, inout Material mat) {
	bool found = false;
	int index = 0;
	float a, b = FLT_MAX;
	float acc_thresh = 0.00001;
	float sums[25];

	int lowerLimit = 0;
	int upperLimit = blob_count;
	//Find range endpoint, make sure it is inside the boundary limit
	
	for (int i = lowerLimit; i < upperLimit; i++) {
		Hit h3;
		h3.t = FLT_MAX;
		bool f = SphereIntersectionWithMod(blobs[i], r, tmin, h3, 2.0f);
		if (f) {
			vec3 mid = (blobs[i].center - PointAtParameter(r, h3.t));
			float t_mid = dot(mid, normalize(r.direction));
			float temp_b = h3.t + t_mid;
			vec3 posb = PointAtParameter(r,temp_b);
			float sum_b = 0.0f;
			float sum_rad = 0.0f;
			for (int i = lowerLimit; i < upperLimit; i++) {
				sum_b += evalF(posb, blobs[i].center, blobs[i].params.x);
				sum_rad += blobs[i].params.x;
			}
			float evalb = (coef - sum_b) / (1.5 * sum_rad);
			if (evalb < 0.0 && temp_b < b) {
				found = true;
				b = temp_b;
			}
		}
		index++;
	}
	
	if (found) {
		a = tmin;
		float c = (a + b) / 2.0;
		int ite = 0;
		
		for(int i = 0; i < 25; i++) {
			sums[i] = 0.0;
		}

		if (b < a) return false;

		while (ite < 200) {
			float evala, evalb, evalc;
			float sum_a = 0.0f;
			float sum_b = 0.0f;
			float sum_c = 0.0f;
			float sum_rad = 0.0f;
			
			vec3 posa = PointAtParameter(r,a);
			vec3 posb = PointAtParameter(r,b);
			vec3 posc = PointAtParameter(r,c);
			

			for (int i = lowerLimit; i < upperLimit; i++) {
				sum_a += evalF(posa, blobs[i].center, blobs[i].params.x);
				sum_b += evalF(posb, blobs[i].center, blobs[i].params.x);
				sum_rad += blobs[i].params.x;
				sums[i] = evalF(posc, blobs[i].center, blobs[i].params.x);
				sum_c += sums[i];
			}
			if (sum_rad == 0.0) { sum_rad = 1.0f; }
			evala = (coef - sum_a) / (1.5 * sum_rad);
			evalb = (coef - sum_b) / (1.5 * sum_rad);
			evalc = (coef - sum_c) / (1.5 * sum_rad);

			if (evala * evalc < 0.0) {
				b = c;
			}
			else  {
				a = c;
			}
			c = (a * evalb - b * evala) / (evalb - evala);


			if (abs(evalc) < acc_thresh) {
				h.t = c;
				h.material = int(spheres[0].params_i.x);
				vec3 normal = vec3(0);
				mat.diffuse_color = vec3(0.0);
				mat.specular_color = vec3(0.0);
				mat.reflective_color = vec3(0.0);
				mat.transparent_color = vec3(0.0);
				mat.params.y = 0.0;
				for (int i = lowerLimit; i < upperLimit; i++) {
					normal += sums[i] * evalN(posc, blobs[i].center, blobs[i].params.x);
					mat.diffuse_color += sums[i] * materials[int(blobs[i].params_i.x)].diffuse_color;
					mat.specular_color += sums[i] * materials[int(blobs[i].params_i.x)].specular_color;
					mat.reflective_color += sums[i] * materials[int(blobs[i].params_i.x)].reflective_color;
					mat.transparent_color += sums[i] * materials[int(blobs[i].params_i.x)].transparent_color;
					mat.params.y += sums[i] * materials[int(blobs[i].params_i.x)].params.y;
				}
				float norm_coef = 1.0/sum_c;
				mat.diffuse_color = norm_coef * mat.diffuse_color;
				mat.specular_color = norm_coef *mat.specular_color;
				mat.reflective_color = norm_coef * mat.reflective_color;
				mat.transparent_color = norm_coef * mat.transparent_color;
				mat.params.y = norm_coef * mat.params.y;
				h.normal = normalize(normal);
				return true;
			}
			
			ite++;
		}
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


bool PlaneIntersection(Plane p, Ray r, float tmin, inout Hit hit) {
	float denom = dot(r.direction, p.normal);
	if (abs(denom) > 0.000001f)
	{
		float t = (p.params.x - dot(r.origin, p.normal)) / dot(r.direction, p.normal);
		vec3 pos = PointAtParameter(r, t);
		if (t > tmin && t < hit.t && p.params.y > abs(pos.x) && p.params.y > abs(pos.z)) {
			hit.t = t;
			hit.material = int(p.params_i.x);
			hit.normal = p.normal;
			return true;
		}
	}
	return false;
}

bool TraceShadow(Ray r, Hit h, float tmin) {
	//Sphere
	bool intersection = false;
	
	for (int i = 0; i < sphere_count; i++) {
		if(spheres[i].params.z < 0.1) {
			continue;
		}
		bool b = SphereIntersection(spheres[i], r, tmin, h);
		if (b && h.t > tmin) {
			return true;
		}
	}
	//Box
	for (int i = 0; i < box_count; i++) {
		bool b = BoxIntersection(boxes[i], r, tmin, h);
		if (b && h.t > tmin) {
			return true;
		}
	}
	//Planes
	for (int i = 0; i < plane_count; i++) {
		bool b = PlaneIntersection(planes[i], r, tmin, h);
		if (b && h.t > tmin) {
			return true;
		}
	}

	//Blobs
	{
		Material blobMaterial;
		bool b = BlobIntersection(r, tmin, h, blobMaterial);
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
	
	Material blobMaterial;
	blobMaterial.params.z = -1.0;

	while (bounces >= 0) {
		
		bool intersection = false;
	
		//Sphere
		for (int i = 1; i < sphere_count; i++) {
			if(spheres[i].params.z < 0.1) {
				continue;
			}
			Hit h2;
			h2.t = FLT_MAX;
			bool b = SphereIntersection(spheres[i], r, tmin, h2);
			if (b && h2.t < hit.t) {
				hit = h2;
				intersection = true;
			}
		}
		
		//Box
		for (int i = 0; i < box_count; i++) {
			Hit h2;
			h2.t = FLT_MAX;
			bool b = BoxIntersection(boxes[i], r, tmin, h2);
			if (b && h2.t < hit.t) {
				hit = h2;
				intersection = true;
			}
		}
		//Planes
		for (int i = 0; i < plane_count; i++) {
			Hit h2;
			h2.t = FLT_MAX;
			bool b = PlaneIntersection(planes[i], r, tmin, h2);
			if (b && h2.t < hit.t) {
				hit = h2;
				intersection = true;
			}
		}

		//Blobs
		{
			Hit h2;
			h2.t = FLT_MAX;
			bool b = BlobIntersection(r, tmin, h2, blobMaterial);
			if (b && h2.t < hit.t) {
				hit = h2;
				blobMaterial.params.z = 1.0;
				intersection = true;
			}
		}
		
				
		if (intersection) {
			Material mat;
			if(blobMaterial.params.z > 0.0) mat = blobMaterial;
			else mat = materials[hit.material];

			answer += ref_col * Shade(hit, r, mat);
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
        return;
	}
    else{
        fragColor = vec4(vUv.x, vUv.y,0,1.0);
    }
}
`;

function getRandomArbitraryVector(min, max) {
	return new THREE.Vector3(Math.random() * (max - min) + min, Math.random() * (max - min) + min, Math.random() * (max - min) + min) 
}

function getRandomArbitrary(min, max) {
	return Math.random() * (max - min) + min
}


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

var animatedSphereIndices = []
var animatedSphereSpeeds = []
var sphereVisible = []

const light_1 = {
    position: new THREE.Vector3(0 ,0, 5),
    direction: new THREE.Vector3( 0, 0, -1 ),
    intensity: new THREE.Vector3( 0.8, 0.8, 0.8 ),
    constant_attenuation: new THREE.Vector3( 0.5, 0.5, 0.5 ),
    linear_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
    quadratic_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
    params : new THREE.Vector3( 1, 1, 1),
}

const light_2 = {
    position: new THREE.Vector3(0, 4, 0),
    direction: new THREE.Vector3( 0, 0, 1 ),
    intensity: new THREE.Vector3( 0.8, 0.8, 0.8 ),
    constant_attenuation: new THREE.Vector3( 0.5, 0.5, 0.5 ),
    linear_attenuation: new THREE.Vector3( 0.5, 0.5, 0.5 ),
    quadratic_attenuation: new THREE.Vector3( 0.05, 0.05, 0.05 ),
    params : new THREE.Vector3( 1, 1, 1),
}

lightsArray.push(light_1)

const sp_1 = {
	center : new THREE.Vector3(0, 1, 0),
    params : new THREE.Vector3(2, 0, 0),
    params_i : new THREE.Vector3(0, -1, 0),
}
animatedSphereIndices.push(spheresArray.length)
spheresArray.push(sp_1)

const material_1 = {
    diffuse_color : new THREE.Vector3(0.8, 0, 0),
    specular_color : new THREE.Vector3(0, 0.7, 0),
    reflective_color : new THREE.Vector3(0.05,0.05,0.05),
    transparent_color : new THREE.Vector3(0.0, 0.0, 0.0),
    params : new THREE.Vector3(0, 4, 0),
}

materialsArray.push(material_1)

const numberOfExtraSpheres = 25
const boxSize = 5

for(var i = 0; i < numberOfExtraSpheres; i++) {
	animatedSphereIndices.push(spheresArray.length)
	
	var sp = {
		center : new THREE.Vector3(0,0,0),
		params : new THREE.Vector3(getRandomArbitrary(0.5,1), 0, 0),
		params_i : new THREE.Vector3(i, -1, 0)
	}

	var vec = getRandomArbitraryVector(-boxSize + 0.01, boxSize - 0.01)
	sp.center = vec

	var mat = {
	    diffuse_color : getRandomArbitraryVector(0,1),
		specular_color : getRandomArbitraryVector(0,1),
		reflective_color : getRandomArbitraryVector(0,0.1),
		transparent_color : new THREE.Vector3(0.0, 0.0, 0.0),
		params : new THREE.Vector3(0, getRandomArbitrary(2,64), 0),
	}

	spheresArray.push(sp)
	materialsArray.push(mat)
}

for(var i = 0; i < spheresArray.length; i++) {
	var vec = getRandomArbitraryVector(-1,1)
	var a = getRandomArbitrary(-2, 2)
	const speed = { dir: vec.normalize(), amplitude: a}
	animatedSphereSpeeds.push(speed);
	sphereVisible.push(1)
}

const box_1 = {
    vmin : new THREE.Vector3(-0.5, 0, -0.5),
    vmax : new THREE.Vector3(0.5, 1, 0.5),
    params_i : new THREE.Vector3(0, -1, 0),
}

const plane_1 = {
    normal : new THREE.Vector3(0, 1, 0),
    params : new THREE.Vector3(-5, 5, 0),
    params_i : new THREE.Vector3(2, -1, 0),
}

const uniforms = {

	skyboxTex : {type: 't', value: null},
    iTime: { value: 0 },
    iResolution:  { value: new THREE.Vector2() },
    
    lights : { 
        value: lightsArray
    },

    spheres : { 
        value: spheresArray
    },

	blobs : { 
        value: spheresArray
    },
    
    boxes : { 
        value: [
            box_1
        ]
    },
    
    planes : { 
        value: [
            plane_1
        ]
    },
    
    
    skyboxIn : { 
        value: {
			vmin_s : new THREE.Vector3(-25, -25, -25),
			vmax_s : new THREE.Vector3(25, 25, 25),
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

	blob_count : {
        value : 0
    },

    box_count : {
        value : 0
    },

    plane_count : {
        value : 0
    },

    has_skybox : {
        value : 1
    },

    samples : {
        value : 4
    },
	
};

const skyboxTexture = new THREE.TextureLoader().load( 
	"images/skyboxDay.png",
);

skyboxTexture.wrapS = THREE.ClampToEdgeWrapping
skyboxTexture.wrapT = THREE.ClampToEdgeWrapping
skyboxTexture.magFilter = THREE.NearestFilter
skyboxTexture.minFilter = THREE.NearestFilter
skyboxTexture.repeat.set( 4, 4 )
uniforms.skyboxTex.value = skyboxTexture

var quad = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms,
        glslVersion: THREE.GLSL3,
        depthWrite: false,
        depthTest: false
    })
  );

scene.add(quad)

function animate(time) {
    requestAnimationFrame(animate)
    controls.update()
	stats.update()
	render(time)
}

function calculatePosition(delta, positionIn, speed, box) {
	const diff = speed.dir.clone().multiplyScalar(speed.amplitude).multiplyScalar(delta);
	var position = positionIn.clone()
	position.add(diff)

	if(position.x < -box) {
		position.x = -box
	}
	else if(position.x > box) {
		position.x = box
	}
	else if(position.y < -box) {
		position.y = -box
	}
	else if(position.y > box) {
		position.y = box
	}
	else if(position.z < -box) {
		position.z = -box
	}
	else if(position.z > box) {
		position.z = box
	}
	return position
}

function calculateSpeed(position, speed, box) {
	var normal = new THREE.Vector3(0,0,0)
	var speedDirection = speed.dir.normalize()
	if(position.x <= -box) {
		normal.set(1,0,0);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	else if(position.x >= box) {
		normal.set(-1,0,0);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	else if(position.y <= -box) {
		normal.set(0,1,0);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	else if(position.y >= box) {
		normal.set(0,-1,0);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	else if(position.z <= -box) {
		normal.set(0,0,1);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	else if(position.z >= box) {
		normal.set(0, 0,-1);
		speedDirection.add(normal.multiplyScalar(-2*normal.dot(speedDirection)))
	}
	speedDirection.normalize()

	return  {
		dir : new THREE.Vector3(speedDirection.x, speedDirection.y, speedDirection.z),
		amplitude : speed.amplitude
	}
		
}

function intersectSphere(sphere1_center, sphere1_radius, sphere1_speed, sphere2_center, sphere2_radius, sphere2_speed) {
	//Masses are in relation to radius^3 -> mass = mass density * 1.333 * pi * r^3
	const center1 = sphere1_center.clone()
	const center2 = sphere2_center.clone()
	const dist = center1.distanceTo(center2);
	const cmp = sphere1_radius + sphere2_radius
	const diff = dist-cmp;
	if(diff < 0.001) {
		const speed1 = sphere1_speed.dir.clone()
		const speed2 = sphere2_speed.dir.clone()
		const r1Cube = Math.pow(sphere1_radius, 3)
		const r2Cube = Math.pow(sphere2_radius, 3)
		const denom = (r1Cube+r2Cube)
		const coef1_1 = (r1Cube-r2Cube) / denom
		const coef1_2 = (2.0 * r2Cube) / denom
		const coef2_1 = (r2Cube-r1Cube) / denom
		const coef2_2 = (2.0 * r1Cube) / denom
		const sphere1_dir = speed1.multiplyScalar(coef1_1).add(speed2.multiplyScalar(coef1_2))
		const sphere2_dir = speed2.multiplyScalar(coef2_1).add(speed1.multiplyScalar(coef2_2))

		const ret1 = {
			dir : sphere1_dir,
			amplitude : (coef1_1 * sphere1_speed.amplitude + coef1_2 * sphere2_speed.amplitude)
		}

		const ret2 = {
			dir : sphere2_dir,
			amplitude : (coef2_1 * sphere2_speed.amplitude + coef2_2 * sphere1_speed.amplitude)
		}
		
		const direction = center2.add(center1.negate()).normalize()		
		return [ret1, ret2, sphere1_center.clone().add(direction.negate().multiplyScalar(Math.abs(diff)))]
	}
	return []
}

function intersectSphereSimple(sphere1_center, sphere1_radius, sphere2_center, sphere2_radius, offset) {
	const center1 = sphere1_center.clone()
	const center2 = sphere2_center.clone()
	const dist = center1.distanceTo(center2);
	const cmp = sphere1_radius + sphere2_radius + offset
	if(dist < cmp) {
		return true;
	}
	else{
		return false;
	}
}

var clock = new THREE.Clock();

function render(time) {
    time *= 0.001;
    const canvas = renderer.domElement
    uniforms.iResolution.value.set(canvas.width, canvas.height)
    uniforms.iTime.value = time;

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

	var delta = clock.getDelta()

	for(var i = 1; i < spheresArray.length; i++) {
		for(var j = i; j < spheresArray.length; j++) {
			var speedsAndPositions = intersectSphere(uniforms.spheres.value[i].center, uniforms.spheres.value[i].params.x, animatedSphereSpeeds[i], uniforms.spheres.value[j].center, uniforms.spheres.value[j].params.x, animatedSphereSpeeds[j])
			if(speedsAndPositions.length > 0) {
				animatedSphereSpeeds[i] = speedsAndPositions[0]
				animatedSphereSpeeds[j] = speedsAndPositions[1]
				uniforms.spheres.value[i].center = speedsAndPositions[2]
			}
		}
	}

	var blobIndex = 0
	var blobs = []
	blobs.push(uniforms.spheres.value[0])
	blobIndex++

	for(var i = 0; i < spheresArray.length; i++) {
		sphereVisible[i] = 1
	}

	for(var i = 1; i < spheresArray.length; i++) {
		var intersects = intersectSphereSimple(uniforms.spheres.value[0].center, uniforms.spheres.value[0].params.x, uniforms.spheres.value[i].center, uniforms.spheres.value[i].params.x, 0.5)
		if(intersects) {
			blobs.push(uniforms.spheres.value[i])
			sphereVisible[i] = 0
			blobIndex++
		}
	}

	for(var i = blobIndex; i < 25; i++) {
		blobs.push(uniforms.spheres.value[0])
	}

	uniforms.blobs.value = blobs
	uniforms.blob_count.value = blobIndex

	for(var i = 1; i < spheresArray.length; i++) {
		animatedSphereSpeeds[i] = calculateSpeed(uniforms.spheres.value[i].center, animatedSphereSpeeds[i], boxSize)
		uniforms.spheres.value[i].center = calculatePosition(delta, uniforms.spheres.value[i].center, animatedSphereSpeeds[i], boxSize)
	}
	sphereVisible[0] = 0
	for(var i = 0; i < spheresArray.length; i++) {
		uniforms.spheres.value[i].params.z = sphereVisible[i]
	}
    renderer.render(scene, camera)

	blobs = []
}

animate()
