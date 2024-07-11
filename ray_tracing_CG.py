import numpy as np
from PIL import Image
import random

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def intersect_sphere(origin, direction, sphere):
    center, radius, color = sphere
    oc = origin - center
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, None, None
    t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
    t = t1 if t1 > 0 else t2
    hit_point = origin + t * direction
    normal = normalize(hit_point - center)
    return True, t, normal

def ray_color(ray_origin, ray_direction, spheres, lights, depth=0):
    if depth >= 3:
        return np.array([0.0, 0.0, 0.0])
    
    closest_t = float('inf')
    closest_sphere = None
    closest_normal = None
    
    for sphere in spheres:
        hit, t, normal = intersect_sphere(ray_origin, ray_direction, sphere)
        if hit and t < closest_t:
            closest_t = t
            closest_sphere = sphere
            closest_normal = normal

    if closest_sphere is None:
        return np.array([0.5, 0.7, 1.0])  # Background color (sky blue)

    hit_point = ray_origin + closest_t * ray_direction
    hit_normal = closest_normal
    hit_color = closest_sphere[2]  # Sphere color
    
    # Lighting
    ambient_light = 0.1
    diffuse_light = 0.0
    specular_light = 0.0
    view_dir = normalize(ray_origin - hit_point)

    for light in lights:
        light_dir = normalize(light - hit_point)
        light_intensity = np.dot(light_dir, hit_normal)
        if light_intensity > 0:
            diffuse_light += light_intensity
            reflect_dir = normalize(2 * np.dot(light_dir, hit_normal) * hit_normal - light_dir)
            specular_light += max(np.dot(view_dir, reflect_dir), 0) ** 32

    color = ambient_light * hit_color + diffuse_light * hit_color + specular_light * np.array([1.0, 1.0, 1.0])
    color = np.clip(color, 0.0, 1.0)
    
    # Reflection
    reflect_dir = normalize(ray_direction - 2 * np.dot(ray_direction, hit_normal) * hit_normal)
    reflect_color = ray_color(hit_point, reflect_dir, spheres, lights, depth + 1)
    color = 0.8 * color + 0.2 * reflect_color

    return color

def render(image_width, image_height, spheres, lights, samples_per_pixel):
    aspect_ratio = image_width / image_height
    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = np.array([0.0, 0.0, 0.0])
    horizontal = np.array([viewport_width, 0.0, 0.0])
    vertical = np.array([0.0, viewport_height, 0.0])
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - np.array([0.0, 0.0, focal_length])

    image = np.zeros((image_height, image_width, 3))

    for j in range(image_height):
        for i in range(image_width):
            color = np.zeros(3)
            for s in range(samples_per_pixel):
                u = (i + random.random()) / (image_width - 1)
                v = (image_height - j - 1 + random.random()) / (image_height - 1)
                direction = lower_left_corner + u * horizontal + v * vertical - origin
                direction = normalize(direction)
                color += ray_color(origin, direction, spheres, lights)
            color /= samples_per_pixel
            image[j, i] = color

        print(f"Progresso: {j / image_height:.2%}")

    image = (255.999 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    return Image.fromarray(image)

if __name__ == "__main__":
    print("Iniciando renderização...")
    image_width = 400
    image_height = 200
    samples_per_pixel = 10
    spheres = [
        (np.array([0.0, 0.0, -1.0]), 0.5, np.array([0.8, 0.3, 0.3])),  # Red sphere
        (np.array([1.0, 0.0, -1.5]), 0.5, np.array([0.3, 0.8, 0.3])),  # Green sphere
        (np.array([-1.0, 0.0, -1.5]), 0.5, np.array([0.3, 0.3, 0.8])),  # Blue sphere
        (np.array([0.0, -100.5, -1.0]), 100.0, np.array([0.8, 0.8, 0.0]))  # Ground sphere (yellow)
    ]
    lights = [
        np.array([5.0, 5.0, 5.0]),
        np.array([-5.0, 5.0, 5.0])
    ]
    image = render(image_width, image_height, spheres, lights, samples_per_pixel)
    image_path = "improved_ray_tracing_output.png"
    image.save(image_path)
    print(f"Imagem gerada e salva como {image_path}")
