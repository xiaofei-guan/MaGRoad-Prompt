import { RoadNetworkData } from '../types';

/**
 * Validates the integrity of road network data (node-edge format only)
 */
export const validateRoadNetwork = (roadNetwork: RoadNetworkData): {
    isValid: boolean;
    errors: string[];
    warnings: string[];
} => {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!roadNetwork) {
        errors.push('Road network data is null or undefined');
        return { isValid: false, errors, warnings };
    }

    if (roadNetwork.type !== 'FeatureCollection') {
        errors.push('Road network must be a GeoJSON FeatureCollection');
    }

    // Validate node-edge format only
    if (!roadNetwork.nodes || !Array.isArray(roadNetwork.nodes)) {
        errors.push('Node-edge format requires a valid nodes array');
    }

    if (!roadNetwork.edges || !Array.isArray(roadNetwork.edges)) {
        errors.push('Node-edge format requires a valid edges array');
    }

    if (roadNetwork.nodes && roadNetwork.edges) {
        const nodeIds = new Set(roadNetwork.nodes.map(n => n.id));
        
        // Check for duplicate node IDs
        if (nodeIds.size !== roadNetwork.nodes.length) {
            errors.push('Duplicate node IDs found');
        }

        // Check edge references
        roadNetwork.edges.forEach((edge, i) => {
            if (!nodeIds.has(edge.source)) {
                errors.push(`Edge ${i} references non-existent source node ${edge.source}`);
            }
            if (!nodeIds.has(edge.target)) {
                errors.push(`Edge ${i} references non-existent target node ${edge.target}`);
            }
            if (edge.source === edge.target) {
                warnings.push(`Edge ${i} creates a self-loop (source === target)`);
            }
        });
    }

    return {
        isValid: errors.length === 0,
        errors,
        warnings
    };
};