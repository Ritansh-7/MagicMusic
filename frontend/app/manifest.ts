import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "CozyBeats Music",
    short_name: "CozyBeats",
    description: "Your sound, evolving.",
    start_url: "/",
    display: "standalone",
    background_color: "#080b14",
    theme_color: "#080b14",
    orientation: "portrait-primary",
    categories: ["music", "entertainment"],
  };
}
