import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls'
import { PLYLoader } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/loaders/PLYLoader'

var parent = document.getElementById("embedded-video-div")

const scene = new THREE.Scene()
scene.add(new THREE.AxesHelper(5))

const light = new THREE.PointLight()
light.position.set(0, 0, 0)
scene.add(light)

const camera = new THREE.PerspectiveCamera(
    60,
    parent.clientWidth / parent.clientHeight,
    0.01,
    1000
)
camera.position.z = 0.25

const renderer = new THREE.WebGLRenderer()
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(parent.clientWidth, parent.clientHeight)
parent.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true

const material = new THREE.MeshPhysicalMaterial({
    color: 0xb2ffc8,
    metalness: 0,
    roughness: 1,
    transparent: false,
    transmission: 0.0,
    side: THREE.DoubleSide,
    clearcoat: 1.0,
    clearcoatRoughness: 0.25,
    vertexColors: THREE.VertexColors 
})

const loader = new PLYLoader()

window.addEventListener('resize', onWindowResize, false)

function onWindowResize() {
    camera.aspect = parent.clientWidth / parent.clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(parent.clientWidth, parent.clientHeight)
    render()
}


function animate() {
    requestAnimationFrame(animate)

    controls.update()

    render()
}

function clearScene() {
    const object = scene.getObjectByName( 'mesh' )
    if(object) {
        object.geometry.dispose()
        object.material.dispose()
        scene.remove( object )
    }
}

function loadRoom() {
    console.log("Loading")
    clearScene()
    loader.load(
    'models/roomSimplified.ply',
    function (geometry) {
        geometry.computeVertexNormals()
        const mesh = new THREE.Mesh(geometry, material)
        mesh.rotateX(Math.PI)
        mesh.name = 'mesh'
        scene.add(mesh)
    },
    (error) => {
        console.log(error)
    }
    )
}

function loadCurvature() {
    clearScene()
    loader.load(
    'models/roomCurvatureSimplified.ply',
    function (geometry) {
        geometry.computeVertexNormals()
        const mesh = new THREE.Mesh(geometry, material)
        mesh.rotateX(Math.PI)
        mesh.name = 'mesh'
        scene.add(mesh)
    },
    (error) => {
        console.log(error)
    }
    )
}

function render() {
    renderer.render(scene, camera)
}

window.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById("room").addEventListener ("click", loadRoom);
    document.getElementById("roomCurvature").addEventListener ("click", loadCurvature);
});

loadRoom()
animate()
