type EventHandler = (...args: any[]) => void;

interface EventBus {
    on(event: string, handler: EventHandler): void;
    off(event: string, handler: EventHandler): void;
    emit(event: string, ...args: any[]): void;
}

function createEventBus(): EventBus {
    const listeners: { [key: string]: EventHandler[] } = {};

    function on(event: string, handler: EventHandler): void {
        if (!listeners[event]) {
            listeners[event] = [];
        }
        listeners[event].push(handler);
    }

    function off(event: string, handler: EventHandler): void {
        if (!listeners[event]) {
            return;
        }
        listeners[event] = listeners[event].filter(h => h !== handler);
    }

    function emit(event: string, ...args: any[]): void {
        if (!listeners[event]) {
            return;
        }
        listeners[event].forEach(handler => handler(...args));
    }

    return {
        on,
        off,
        emit,
    };
}

// Export a singleton instance
export const globalEventBus = createEventBus(); 