export class SharedAsyncResource<T> {
  private promise: Promise<T> | null = null;

  public get(loader: () => Promise<T>): Promise<T> {
    if (!this.promise) this.promise = loader();
    return this.promise;
  }
}
